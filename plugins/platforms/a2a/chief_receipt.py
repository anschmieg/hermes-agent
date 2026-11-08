"""Fail-closed read-only Chief inputId receipt contract after a remote-stop crash.

The caller must supply an authenticated *read-only* Chief owner transport with
lookup_input(input_id, peer, agent_slug, tenant, context_id). A synthetic
receipt proves only protocol shape, never Chief authority or backend stop.
This module deliberately cannot resend, cancel or settle a Hermes task.
"""
from __future__ import annotations

import math
import time
from typing import Any


def lookup_stop_receipt(ledger: Any, task_id: str, *, peer: str, agent_slug: str,
                        tenant: str, input_id: str, reader: Any = None) -> dict | None:
    """Observe exact owner inputId without asserting cancellation or mutating state.

    The reader's caller is responsible for authenticating the authoritative
    Chief owner endpoint. No authenticated Chief adapter is implemented here.
    """
    if (not isinstance(task_id, str) or not task_id or not isinstance(peer, str)
            or not peer or not isinstance(input_id, str) or not input_id
            or not isinstance(agent_slug, str) or not isinstance(tenant, str)):
        return None
    rec = ledger.get(task_id)
    if (not isinstance(rec, dict) or rec.get("task_id") != task_id
            or rec.get("peer") != peer or rec.get("agent_slug") != agent_slug
            or rec.get("tenant") != tenant or rec.get("input_id") != input_id
            or rec.get("state") != "TASK_STATE_WORKING"
            or rec.get("effect_state") != "uncertain"
            or rec.get("stop_attempt_state") != "uncertain"):
        return None
    attempted = rec.get("stop_attempted_at")
    if (type(attempted) not in (int, float) or not math.isfinite(attempted)
            or not isinstance(rec.get("context_id"), str) or not rec["context_id"]):
        return None
    lookup = getattr(reader, "lookup_input", None)
    if not callable(lookup):
        return None
    try:
        receipt = lookup(input_id=input_id, peer=peer, agent_slug=agent_slug,
                         tenant=tenant, context_id=rec["context_id"])
    except Exception:
        # Unavailable transport is unknown, never permission to send or stop again.
        return None
    if type(receipt) is not dict:
        return None
    expected = {"inputId": input_id, "peer": peer, "agent": agent_slug,
                "tenant": tenant, "contextId": rec["context_id"]}
    if any(type(receipt.get(k)) is not str or receipt[k] != v
           for k, v in expected.items()):
        return None
    updated = receipt.get("updatedAt")
    if (type(updated) not in (int, float) or not math.isfinite(updated)
            or updated < attempted or updated > time.time() + 60):
        return None
    status = receipt.get("status")
    if type(status) is not str or status not in {"queued", "running", "terminal"}:
        return None
    # Even status=terminal + outcome=canceled + stopConfirmed=true does NOT
    # prove a backend stop. Never produce terminal/canceled Hermes state.
    return {"task_id": task_id, "input_id": input_id, "owner_status": status,
            "task_state": "TASK_STATE_WORKING", "stop_confirmed": False}
