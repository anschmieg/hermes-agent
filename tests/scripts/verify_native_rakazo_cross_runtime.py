"""Real Rakazo TypeScript A2AClient -> Hermes native JSON-RPC HTTP test.

Requires an independently checked-out Rakazo tree containing
``packages/adapters/src/a2a-native-hermes.e2e.ts`` and its Node dependencies.
No real credentials, gateway, database, or running server are contacted.

    python tests/scripts/verify_native_rakazo_cross_runtime.py /path/to/rakazo
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_native_a2a_durable_stdlib import NativeHttpAcceptance  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: verify_native_rakazo_cross_runtime.py /path/to/rakazo", file=sys.stderr)
        return 2
    rakazo = Path(sys.argv[1]).resolve()
    client_test = rakazo / "packages/adapters/src/a2a-native-hermes.e2e.ts"
    if not client_test.is_file():
        print(f"Missing real Rakazo TypeScript acceptance client: {client_test}", file=sys.stderr)
        return 2

    fixture = NativeHttpAcceptance("test_return_immediately_persists_task_then_late_reply")
    fixture.setUp()
    env = os.environ.copy()
    env.update({
        "HERMES_A2A_TEST_URL": fixture.base,
        "HERMES_A2A_TEST_MODE": "isolated-native-http-fake-agent",
    })
    print("Hermes native A2A HTTP fixture on ephemeral loopback port", flush=True)
    try:
        result = subprocess.run(
            ["pnpm", "exec", "tsx", "packages/adapters/src/a2a-native-hermes.e2e.ts"],
            cwd=rakazo, env=env, timeout=45, check=False,
        )
        print(f"RAKAZO_NATIVE_A2A_EXIT={result.returncode}", flush=True)
        return result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Cross-runtime test failed to execute: {exc}", file=sys.stderr)
        return 2
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    sys.exit(main())
