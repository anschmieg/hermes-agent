# A2A — Agent-to-Agent protocol for Hermes

Talk to other agents, and let other agents talk to you, over the open
[A2A protocol](https://a2a-protocol.org) **v1.0**. Works with any A2A-compliant
peer (another Hermes, LangChain, CrewAI, Google ADK, OpenClaw, …). Stdlib only —
no `a2a-sdk` dependency.

## Enable

```bash
hermes gateway setup      # pick A2A, or:
```

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900

# peers you want to call (outbound):
a2a_agents:
  researcher:
    url: "http://localhost:9999"
    auth: { type: bearer, token: "sk-..." }
    timeout: 120
    capabilities: [web_search, research]
```

## Outbound — call other agents

The agent gets five tools:

- `a2a_discover(url)` — what can this agent do?
- `a2a_call(agent, message, context_id?)` — send it a task, get the reply.
- `a2a_list()` — configured peers, saved conversations, metrics.
- `a2a_history(context_id)` — recall a saved A2A conversation.
- `a2a_orchestrate(capability, message, mode?)` — fan-out a task to every
  peer advertising a capability (`all` / `first` / `best`).

## Inbound — be callable

When the `a2a` platform is enabled, Hermes serves a v1.0 Agent Card at
`http://<host>:<port>/.well-known/agent-card.json` (the legacy
`/.well-known/agent.json` path is also answered for pre-1.0 clients) and
accepts JSON-RPC
`message/send`, `message/stream` (SSE), `tasks/get|list|cancel|subscribe`,
and push notification configs (inline or via
`tasks/pushNotificationConfig/create`). Incoming tasks are injected into your
**live** agent session — the same agent that's talking to you, with full
memory — and the reply is returned over A2A. Completed tasks stay queryable
via `tasks/get`.

### Durable task admission and recovery

- Incoming A2A `message.messageId` is required. A SQLite journal in the active
  Hermes profile records each `(authenticated peer, messageId)` reservation
  **before** dispatch; concurrent/restarted requests with the same payload and
  scope receive the original `taskId`. Reuse with different payload or tenant
  returns an idempotency conflict (`-32053`).
- A second durable transaction marks dispatch **uncertain before** invoking a
  gateway/owner. A `SIGKILL`, timeout, disconnected SSE client, or retry never
  synthesizes a FAILED result or sends a second copy of uncertain work.
- `configuration.returnImmediately: true` returns the committed task id without
  waiting for inference; call `GetTask` or `SubscribeToTask` for settlement.
  A late gateway reply updates the journal even if the original HTTP client
  has disconnected. Terminal outcomes are immutable on subsequent retries.
- `CancelTask` currently **rejects** active work (`-32002`) because no backend
  stop acknowledgement exists. A `TASK_STATE_CANCELED` result must never be
  fabricated while execution may still be running.
- A dispatch claimed before a crash may remain `TASK_STATE_WORKING` until an
  authoritative outcome is reconciled. **Do not automatically resubmit** such
  work or mistake an unresolved task for failed work. Exactly-once external
  effects require the actual backend's own idempotent admission/settlement.
- For transports that lose the original `SendMessage` response, Hermes also
  offers **`GetTaskByMessageId`**, a nonstandard read-only JSON-RPC extension:
  `params: {"messageId": "original-id"}`. It returns the original v1.0 Task
  only for the authenticated peer and routed agent/tenant. Rakazo uses this to
  reconcile an uncertain send without issuing another `SendMessage`.

Verification without live endpoints:

```bash
python tests/scripts/verify_native_a2a_peer_scope_stdlib.py
python tests/scripts/verify_native_a2a_durable_stdlib.py
```

The latter executes the real native HTTP handler and SQLite ledger with a fake
gateway, including multiprocess races and a real SIGKILL. It is **not** a live
Chief↔Hermes↔Rakazo acceptance test.

## Security

- **No token ⇒ localhost only.** The server binds `127.0.0.1` and refuses to
  widen unless you configure a token *and* set `A2A_HOST`.
  Without a token it also refuses browser-originated POSTs (`Origin`,
  cross-site `Sec-Fetch-Site`, a missing or non-JSON `Content-Type`).
- **No token ⇒ loopback `Host` only.** Every route refuses a `Host` that is
  not a loopback name (`localhost`, `127.x.x.x`, `[::1]`) or the
  `A2A_PUBLIC_URL` host, so a DNS-rebound page cannot reach it. Only set
  `A2A_PUBLIC_URL` to a name you control.
- **Per-peer tokens**: `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` gives each
  remote agent its own credential; that authenticated name (never anything
  in the request body) drives rate limiting, trust, and audit.
- Inbound text — including `/`-prefixed text — is run through
  prompt-injection filters and framed as untrusted peer input; remote peers
  cannot invoke operator slash commands.
- Outbound text is scrubbed of credential-shaped strings.
- Push callbacks are SSRF-guarded and HMAC-SHA256 signed (`X-A2A-Signature`).
- Every exchange is logged to `~/.hermes/a2a_audit.jsonl`.
- Conversations persist to `~/.hermes/a2a_conversations/` — they survive context
  compaction and restarts (`a2a_history` recalls them).

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `A2A_PEER_TOKENS` | _(unset)_ | Per-peer credentials `name:token,…` (preferred). |
| `A2A_BEARER_TOKEN` | _(unset)_ | Shared token; identity falls back to caller IP. |
| `A2A_HOST` | `127.0.0.1` | Bind host. Only widens with a token set. |
| `A2A_PORT` | `9900` | Inbound port. |
| `A2A_AGENT_NAME` | hostname-derived | Name on the Agent Card. |
| `A2A_PUBLIC_URL` | _(unset)_ | Routable URL advertised on the card (reverse proxies). |
| `A2A_TRUSTED_PEERS` | _(unset)_ | Allow-list of authenticated identities. Required for a non-loopback bind (with a token set) unless `A2A_ALLOW_ALL_USERS=true`; otherwise dispatch is refused. |
| `A2A_ALLOW_ALL_USERS` | `false` | Allow any authed peer (dev only). |
| `A2A_RATE_LIMIT` | `60` | Requests/minute per identity. |
| `A2A_MAX_PINGPONG_TURNS` | `5` | Anti-loop turn cap per context (max 20). |
| `A2A_REPLY_TIMEOUT` | `300` | Seconds to wait for the agent's reply; the orphan sweep never fails a task before this window (floor 300s) or while a request still waits on it. |
| `A2A_TASK_LEDGER_PATH` | `<profile HERMES_HOME>/a2a_native_tasks.sqlite3` | Per-profile SQLite WAL task/idempotency journal (owner-only file; synchronous FULL). Back it up with the profile. |
| `A2A_PUSH_SECRET` | bearer token | HMAC secret for push signing. |
| `A2A_ADVERTISED_TOOLSETS` | all registered | Restrict skills on the Agent Card. |

See `DESIGN.md` for architecture and the requirement-tracing table.
