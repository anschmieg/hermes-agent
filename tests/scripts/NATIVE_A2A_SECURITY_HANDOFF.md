# Native Hermes A2A security handoff — 2026-11-07

Branch: `feature/native-a2a-durable-20261107`. Base checkpoint `f6ed549084e6e2ad9087fe2d7cba2051d0e9f8a5`. Most recent reviewed code/test checkpoint before this note: `75bcca4df5a28442674b29b424042380f6766343`.

## Source-confirmed findings fixed on feature branch

1. **P0 cross-peer/tenant context collision.** Previously the network caller's `message.contextId` became the gateway `chat_id`, reply FIFO key, and conversation JSONL file key. Two authenticated peers using the same context string could therefore share session/reply/history state. Now the *internal* context key is SHA-256 of `(agent slug, tenant, authenticated peer, public contextId)`, while canonical public A2A receipts retain the original contextId. Inbound logs, forwarded owner contexts, turn limits, pending reply callbacks, and outbound log settlement use this scoped key.
2. **Shared bearer identity wrongly depended on source IP.** It now maps to the explicitly undifferentiated principal `bearer:shared`, not proxy IP. A shared bearer is **not** multiple authenticated peers: use distinct `A2A_PEER_TOKENS` for Chief, Rakazo, and other independent clients. Recheck `A2A_TRUSTED_PEERS` when migrating any old IP-based allowlist.
3. **Non-root served-agent tenant route missing explicit access control.** HTTP JSON-RPC now denies remote access to a non-root served agent without its `allowed_peers` list, unless `A2A_ALLOW_ALL_USERS` was explicitly enabled. A URL `tenant` value is routing metadata, never authorization. Example per-agent config: `"allowed_peers": ["chief", "rakazo"]`. Root agent and localhost-only trust domains retain their prior defaults.

## Evidence

Executed **locally in isolated ChatCodex checkout**, against the real adapter, ledger and stdlib HTTP server with fake inference only:

- `python tests/scripts/verify_native_a2a_context_isolation.py` — **20 passed**; new adversarial test group was red before patch (exit 1) and green afterward. Tests both peers using the same contextId, same peer across two tenants, history/reply isolation, remote 403 on cross-tenant route, and shared-bearer identity.
- `python tests/scripts/verify_native_a2a_durable_stdlib.py` — **15 passed**, including real SIGKILL after dispatch claim, separate-process concurrent admission, restart, timeout, late terminal and native HTTP.
- `python tests/scripts/verify_native_a2a_peer_scope_stdlib.py` — **4 passed**.
- `python -m py_compile plugins/platforms/a2a/{adapter,security,protocol,ledger}.py` and `git diff --check` — success.

`tests/scripts/verify_native_rakazo_cross_runtime.py` could **not** run in this owner workspace: no Node/pnpm runtime, and a separate Rakazo checkout is needed. Rakazo's current branch contains a TS test client but its passing execution with this new Hermes SHA has **not** been verified here. No production endpoints, live bearer tokens, or real unsettled requests were touched.

## Remaining release gates — **NO-GO**

- Separate adversarial review of the committed source, plus CI for this commit (no status checks seen).
- Run the real Rakazo TypeScript `A2AClient` against an ephemeral **current-branch Hermes** Python HTTP fixture with fake inference and negative tenant/peer tests.
- Prove the Rakazo disposable PostgreSQL ExternalEffect and lease-fencing assertions (placeholder `expect(true)` is invalid evidence).
- Actual authorized Chief/Rakazo ↔ Hermes task receipt, persisted message and restart/cancel semantics. Never resubmit an uncertain live task.
- Confirm production credentials use distinct per-peer tokens and explicit served-agent ACLs. Shared bearer remains a single trust domain.
- No merge/deploy without explicit operator approval.

Coordinator should read this note as an evidence checkpoint, not a production GO.
