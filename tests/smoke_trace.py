"""Simple CI smoke test to validate Langfuse client availability and auth.

Run this in CI to ensure the installed langfuse SDK is present and the client can authenticate.
"""
import sys

try:
    from langfuse import get_client
except Exception as e:
    print(f"langfuse import failed: {e}")
    sys.exit(2)


def main() -> int:
    try:
        client = get_client()
        version = getattr(client, "__version__", None)
        print("langfuse client loaded; version:", version)
        if hasattr(client, "auth_check"):
            ok = client.auth_check()
            print("auth_check:", ok)
            return 0 if ok else 3
        return 0
    except Exception as e:
        print("client init failed:", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
