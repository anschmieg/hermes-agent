"""Crash-safe, cross-process A2A task admission and task receipts (stdlib SQLite).

The journal is authoritative.  A reservation is committed before any backend
dispatch, and a dispatch claim is committed *before* handing work to the agent.
Consequently a process crash can leave an uncertain WORKING task, but cannot
silently admit the same ``(authenticated_peer, messageId)`` a second time.
Do not treat an uncertain task as FAILED or retry the side effect automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any


class IdempotencyConflict(ValueError):
    """A peer reused its messageId with different payload, agent, or tenant."""


class DurableTaskLedger:
    """A per-profile transaction journal, safe for concurrent processes.

    SQLite's BEGIN IMMEDIATE serializes reservation and dispatch claims across
    independent processes; journal_mode=WAL and synchronous=FULL make each
    committed transaction recoverable after a process crash.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Create owner-only. A pre-existing DB is not chmod'd (operator-owned).
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        conn = self._db()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS a2a_tasks (
                task_id TEXT PRIMARY KEY,
                peer TEXT NOT NULL,
                message_id TEXT,
                payload_hash TEXT,
                record TEXT NOT NULL,
                UNIQUE(peer, message_id)
            )""")
        finally:
            conn.close()

    def _db(self):
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @staticmethod
    def fingerprint(payload: Any, agent_slug: str, tenant: str, context_id: str) -> str:
        body = {"payload": payload, "agent": agent_slug, "tenant": tenant, "context": context_id}
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _record(task_id: str, context_id: str, peer: str, agent_slug: str, tenant: str,
                message_id: str, input_id: str) -> dict:
        return {
            "task_id": task_id, "context_id": context_id, "peer": peer,
            "agent_slug": agent_slug, "tenant": tenant, "message_id": message_id,
            "input_id": input_id, "state": "TASK_STATE_SUBMITTED", "reply": "",
            "created_at": time.time(),
            "created_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "push_url": "", "push_config_id": "", "effect_state": "reserved",
        }

    def reserve(self, *, peer: str, message_id: str, payload: Any, context_id: str,
                agent_slug: str = "", tenant: str = "", input_id: str = "") -> tuple[dict, bool]:
        """Atomically reserve messageId; True on replay with identical scope/body."""
        if not peer or not message_id or not context_id:
            raise ValueError("peer, messageId and contextId must not be empty")
        digest = self.fingerprint(payload, agent_slug, tenant, context_id)
        conn = self._db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload_hash, record FROM a2a_tasks WHERE peer=? AND message_id=?",
                               (peer, message_id)).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise IdempotencyConflict("messageId already reserved with different body or scope")
                rec = json.loads(row[1])
                conn.commit()
                return rec, True
            rec = self._record("task-" + uuid.uuid4().hex[:16], context_id,
                               peer, agent_slug, tenant, message_id, input_id)
            conn.execute("INSERT INTO a2a_tasks (task_id, peer, message_id, payload_hash, record) VALUES (?, ?, ?, ?, ?)",
                         (rec["task_id"], peer, message_id, digest, json.dumps(rec, separators=(",", ":"))))
            conn.commit()
            return rec, False
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def write_new(self, rec: dict) -> None:
        """Persist a task without messageId (legacy internal construction only)."""
        conn = self._db()
        try:
            conn.execute("INSERT INTO a2a_tasks (task_id, peer, message_id, payload_hash, record) VALUES (?, ?, NULL, NULL, ?)",
                         (rec["task_id"], rec["peer"], json.dumps(rec, separators=(",", ":"))))
        finally:
            conn.close()

    def update(self, task_id: str, *, claim: bool = False, allow_terminal: bool = False,
               require_reserved: bool = False,
               **fields) -> dict | None:
        """Single durable state transition; `claim` is a once-only dispatch gate.

        Returns None for a lost claim, missing task, or attempted mutation of a
        terminal state. A claimed task remains WORKING after crash/restart.
        """
        # Receipt identity is minted at admission, not mutable through task
        # state transitions. Otherwise a stale/cross-tenant worker could
        # retarget an idempotency record or its original Chief inputId.
        immutable = {"task_id", "context_id", "peer", "agent_slug", "tenant",
                     "message_id", "input_id", "created_at", "created_iso",
                     "stop_attempt_state"}
        if immutable.intersection(fields):
            return None
        conn = self._db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT record FROM a2a_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                conn.rollback()
                return None
            rec = json.loads(row[0])
            if not allow_terminal and rec["state"] in {
                "TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"
            }:
                conn.rollback()
                return None
            if require_reserved and rec.get("effect_state") != "reserved":
                # A concurrent replay may already have claimed dispatch.
                # Validation rejection cannot settle a live side effect.
                conn.rollback()
                return None
            if claim:
                if rec.get("effect_state") != "reserved":
                    conn.rollback()
                    return None
                fields.update(state="TASK_STATE_WORKING", effect_state="uncertain")
            rec.update(fields)
            conn.execute("UPDATE a2a_tasks SET record=? WHERE task_id=?",
                         (json.dumps(rec, separators=(",", ":")), task_id))
            conn.commit()
            return rec
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def claim_stop_attempt(self, task_id: str, *, peer: str, agent_slug: str,
                           tenant: str, input_id: str) -> dict | None:
        """Fence a remote stop attempt before any backend call.

        This is NOT stop confirmation: after a crash the marker is uncertain
        and the stop must not be sent again. Reconcile only by authenticated,
        read-only lookup of the original owner inputId; a terminal CANCELED
        receipt additionally requires independent backend-confirmed stop.
        """
        if not peer or not input_id:
            return None
        conn = self._db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT record FROM a2a_tasks WHERE task_id=?",
                               (task_id,)).fetchone()
            if row is None:
                conn.rollback()
                return None
            rec = json.loads(row[0])
            if (rec.get("peer") != peer or rec.get("agent_slug", "") != agent_slug
                    or rec.get("tenant", "") != tenant or rec.get("input_id") != input_id
                    or rec.get("state") != "TASK_STATE_WORKING"
                    or rec.get("effect_state") != "uncertain"
                    or rec.get("stop_attempt_state")):
                conn.rollback()
                return None
            rec["stop_attempt_state"] = "uncertain"
            rec["stop_attempted_at"] = time.time()
            conn.execute("UPDATE a2a_tasks SET record=? WHERE task_id=?",
                         (json.dumps(rec, separators=(",", ":")), task_id))
            conn.commit()
            return rec
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def claim_push(self, task_id: str) -> str:
        """Atomically claim a pending callback, avoiding cross-process double push."""
        conn = self._db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT record FROM a2a_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                conn.rollback()
                return ""
            rec = json.loads(row[0])
            url = rec.get("push_url") or ""
            if url:
                rec["push_url"] = ""
                rec["push_config_id"] = ""
                conn.execute("UPDATE a2a_tasks SET record=? WHERE task_id=?",
                             (json.dumps(rec, separators=(",", ":")), task_id))
            conn.commit()
            return url
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get(self, task_id: str) -> dict | None:
        conn = self._db()
        try:
            row = conn.execute("SELECT record FROM a2a_tasks WHERE task_id=?", (task_id,)).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    def get_by_message(self, peer: str, message_id: str) -> dict | None:
        """Read-only recovery lookup keyed by trusted peer + original messageId."""
        conn = self._db()
        try:
            row = conn.execute("SELECT record FROM a2a_tasks WHERE peer=? AND message_id=?",
                               (peer, message_id)).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    def records(self) -> list[dict]:
        conn = self._db()
        try:
            return [json.loads(r[0]) for r in conn.execute("SELECT record FROM a2a_tasks ORDER BY rowid")]
        finally:
            conn.close()
