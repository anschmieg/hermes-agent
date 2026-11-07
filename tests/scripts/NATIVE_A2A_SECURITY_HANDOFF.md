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

## 2026-11-07 23:24 CET — independent next gate (owner run)

**Hermes owner code/test HEAD:** `9b65eb8834fdfe953ca3074e76d8332601dc2fa6` before this handoff-note commit. This feature includes HTTP negative Content-Length rejection, and portable cross-runtime test harness; the cross-tenant context/session isolation patch at `3a1996c` remains intact.

- **Actual TypeScript client ↔ native Hermes HTTP PASS (exit 0)** using *current feature-branch Hermes* adapter/security/ledger/protocol mirrored into a dedicated ChatCodex checkout, and Rakazo's real `packages/adapters/src/a2a-native-hermes.e2e.ts` / `A2AClient` from `anschmieg/rakazo-a2a-handoff` feature/native-a2a-bots-spike (baseline `f44a15c`), Node 24 + tsx. The Python runner `tests/scripts/verify_native_rakazo_cross_runtime.py` launches the real native HTTP fixture on an ephemeral loopback port with fake inference. Output `RAKAZO↔HERMES NATIVE HTTP E2E PASS`; `CROSS_RUNTIME_EXIT=0`. Script supports `HERMES_A2A_NODE` and `HERMES_A2A_TSX_CLI` when pnpm is unavailable. Real code paths checked: SendMessage, duplicate idempotency, GetTask, GetTaskByMessageId, Bob forbidden, mutated retry rejected, final ACK.
- **New HTTP framing flaw:** Negative `Content-Length` passed into `rfile.read(-1)`, potentially blocking a persistent HTTP worker and reading unbounded body. Adapter now rejects it with HTTP 400 *before* reading any body. `tests/scripts/verify_native_a2a_http_framing.py` verifies `-1`, `-1024`, >1MB, normal JSON-RPC, and no task admitted for malformed frames.
- **Retested isolated native real-code suites:** framing **3 passed**, context/security **20 passed**, durable crash/recovery **15 passed** including real SIGKILL and process races, peer scope **4 passed**. Compilation and `git diff --check` both exit 0. Counts are distinct; not mock-only production acceptance. No real credentials, production requests, replay of uncertain tasks, merge or deployment.

**NO-GO until:** live authorized Chief/Rakazo end-to-end persisted Bot→Thread→Run acceptance, real disposable PostgreSQL ExternalEffect/lease-fencing negative proof, backend-confirmed stop for cancel, credentials and routed ACL validation at deployment, and independent review/CI against this SHA. Cross-runtime integration proves the real TS HTTP client, not production Chief inference. Coordinator notification action returned 404; this checked-in note is the durable handoff for coordinator consumption.

## 2026-11-08 ~00:40 CET — push-callback SSRF/redirect hardening (new independent gate)

**New finding:** An authenticated remote A2A peer could register an arbitrary push callback hostname. The prior `is_safe_callback_url` only checked URL spelling/literal IP; `urllib.request.urlopen` independently resolved that hostname and automatically followed HTTP redirects. A hostname resolving to an RFC1918/loopback address, or a redirect to one, could reach internal services. Source review found this on `fb25fe3` (adapter `_send_push_notification` and security callback validation). This is distinct from the already-fixed peer context isolation and Content-Length framing bugs.

**Fix now committed on FEATURE branch:** `e532dbd` security resolver, `d26fa65` outbound transport, `23b726d` adversarial test suite. `security.resolve_safe_callback_target` resolves and validates **every** DNS answer, rejects private/link-local/reserved/mixed addresses and URL credentials, and returns the vetted numeric IP. Outgoing `http.client` connects to that IP directly with the original Host and HTTPS SNI/certificate verification, **without** urllib implicit DNS re-lookup, proxy use or redirect following. Localhost push remains available only in deliberately localhost-only mode. Failed DNS resolution refuses delivery. Existing syntactic `is_safe_callback_url` API remains unchanged for compatibility; **only** the pinned target resolver is appropriate for performing egress.

**Red/green proof (isolated ChatCodex Hermes source mirror of feature branch):** Before the fix, `python tests/scripts/verify_native_a2a_push_egress.py` failed all 6 initial adversarial tests (the new resolver did not exist and unpinned urllib was called). After the fix, the expanded suite reports **8 passed, exit 0**, covering private DNS, mixed private/public answers, resolver failure, URL userinfo, DNS pinning/no second lookup, HTTPS SNI/SSL context, actual local ephemeral HTTP POST on 127.0.0.1 and actual HTTP 302 not followed. No production endpoints or credentials accessed. After source changes, `verify_native_a2a_context_isolation.py` **20 passed (exit 0)**, `verify_native_a2a_durable_stdlib.py` **15 passed (exit 0, real SIGKILL and process race)**, `verify_native_a2a_peer_scope_stdlib.py` **4 passed**, and `verify_native_a2a_http_framing.py` **3 passed**; `py_compile` succeeded. The broad `pytest` collection **could not run** in this isolated workspace: `No module named pytest`. The new HTTPS flow is tested with a mocked TLS socket; no actual public certificate handshake was attempted.

**Release decision remains NO-GO** pending production-credential/ACL review, real authorized Chief and Rakazo Bot→Thread→Run persisted acceptance, Rakazo disposable PostgreSQL ExternalEffect/lease tests, backend-confirmed cancellation and independent CI. Pinning is intended to prevent DNS rebinding even when the original DNS name changes between validation and connection; do not revert to generic `urllib.urlopen`. This branch is not merged or deployed.
