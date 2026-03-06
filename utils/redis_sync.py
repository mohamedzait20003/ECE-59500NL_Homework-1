"""
redis_sync.py — Redis-based synchronisation for the two-laptop debate.

Instead of relying on microphone-to-speaker audio capture between laptops,
both sides connect to the same Redis instance and exchange debate text via
blocking list operations (RPUSH / BLPOP).  This guarantees:

  • 100 % accurate text delivery (no STT noise)
  • Proper turn-taking synchronisation (blocking reads)
  • Works over any network (LAN, Wi-Fi, even the internet)

TTS still plays on each laptop for the audience — Redis only carries
the *text* so each side knows what the opponent said.

Channels (Redis list keys):
  debate:{session}:to_trump     — messages for the Trump terminal
  debate:{session}:to_biden     — messages for the Biden terminal
  debate:{session}:ready        — both sides push "ready" here to handshake

Usage from a debate script:

    sync = DebateSync("trump", redis_host="192.168.1.50")
    sync.wait_for_opponent()          # blocks until both sides are ready
    sync.send(text)                   # pushes text to opponent's queue
    opp = sync.receive(timeout=120)   # blocks until opponent sends text
    sync.close()
"""

import os
import sys
import time

try:
    import redis as _redis_lib
    _redis_ok = True
except ImportError:
    _redis_lib = None
    _redis_ok = False


# ══════════════════════════════════════════════════════════════════════════════
# Public class
# ══════════════════════════════════════════════════════════════════════════════

class DebateSync:
    """Thin wrapper around Redis for debate turn synchronisation."""

    def __init__(
        self,
        persona: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = "",
        session: str = "default",
    ):
        if not _redis_ok:
            raise ImportError(
                "The 'redis' package is required for Redis sync. "
                "Install it with:  pip install redis"
            )

        self.persona  = persona.lower()          # "trump" or "biden"
        self.opponent = "biden" if self.persona == "trump" else "trump"
        self.session  = session

        # Key names
        self._key_to_self = f"debate:{session}:to_{self.persona}"
        self._key_to_opp  = f"debate:{session}:to_{self.opponent}"
        self._key_ready   = f"debate:{session}:ready"

        # Connect
        pwd = redis_password or os.getenv("REDIS_PASSWORD", "")
        self._r = _redis_lib.Redis(
            host=redis_host,
            port=redis_port,
            password=pwd or None,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=None,        # BLPOP handles its own timeout
        )

        # Verify connection
        try:
            self._r.ping()
            print(
                f"[redis_sync] Connected to Redis at {redis_host}:{redis_port} "
                f"(session={session!r}, persona={self.persona})",
                file=sys.stderr,
            )
        except _redis_lib.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach Redis at {redis_host}:{redis_port} — {exc}\n"
                "Make sure Redis is running and reachable from this machine."
            ) from exc

    # ── Handshake ──────────────────────────────────────────────────────────

    def wait_for_opponent(self, timeout: int = 300) -> bool:
        """
        Both sides call this. Each pushes 'ready', then blocks until the
        other side also pushes 'ready'.  Returns True when both are ready.
        """
        # Clear stale readiness signals
        self._r.delete(self._key_ready)
        time.sleep(0.3)   # give the other side time to also clear

        self._r.rpush(self._key_ready, self.persona)
        print(
            f"[redis_sync] {self.persona.upper()} signalled ready — "
            f"waiting for {self.opponent.upper()} …",
            file=sys.stderr,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            members = self._r.lrange(self._key_ready, 0, -1)
            if self.opponent in members:
                print(
                    f"[redis_sync] {self.opponent.upper()} is ready — "
                    f"handshake complete!",
                    file=sys.stderr,
                )
                return True
            time.sleep(0.5)

        print("[redis_sync] Timed out waiting for opponent.", file=sys.stderr)
        return False

    # ── Send / Receive ─────────────────────────────────────────────────────

    def send(self, text: str, msg_type: str = "response") -> None:
        """Push a message to the opponent's queue."""
        payload = f"{msg_type}|{text}"
        self._r.rpush(self._key_to_opp, payload)
        print(
            f"[redis_sync] SENT → {self.opponent.upper()} "
            f"({msg_type}, {len(text)} chars)",
            file=sys.stderr,
        )

    def receive(self, timeout: int = 120) -> tuple[str, str]:
        """
        Block until a message arrives on our own queue.
        Returns (msg_type, text).  On timeout returns ("timeout", "").
        """
        print(
            f"[redis_sync] Waiting for message from {self.opponent.upper()} "
            f"(timeout={timeout}s) …",
            file=sys.stderr,
        )
        result = self._r.blpop(self._key_to_self, timeout=timeout)
        if result is None:
            print("[redis_sync] Receive timed out.", file=sys.stderr)
            return ("timeout", "")

        _key, payload = result
        if "|" in payload:
            msg_type, text = payload.split("|", 1)
        else:
            msg_type, text = "response", payload

        print(
            f"[redis_sync] RECV ← {self.opponent.upper()} "
            f"({msg_type}, {len(text)} chars)",
            file=sys.stderr,
        )
        return (msg_type, text)

    # ── Cleanup ────────────────────────────────────────────────────────────

    def flush_session(self) -> None:
        """Delete all keys for this session (call once at debate end)."""
        for key in self._r.scan_iter(f"debate:{self.session}:*"):
            self._r.delete(key)
        print("[redis_sync] Session keys flushed.", file=sys.stderr)

    def close(self) -> None:
        """Clean up the connection."""
        try:
            self._r.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Redis sync smoke-test")
    p.add_argument("--persona", required=True, choices=["trump", "biden"])
    p.add_argument("--host",    default="localhost")
    p.add_argument("--port",    type=int, default=6379)
    args = p.parse_args()

    sync = DebateSync(args.persona, redis_host=args.host, redis_port=args.port)
    ok = sync.wait_for_opponent(timeout=60)
    if not ok:
        sys.exit(1)

    if args.persona == "trump":
        sync.send("Hello from Trump!")
        _, msg = sync.receive()
        print(f"Trump heard: {msg}")
    else:
        _, msg = sync.receive()
        print(f"Biden heard: {msg}")
        sync.send("Hello from Biden!")

    sync.close()
    print("Smoke-test passed.")
