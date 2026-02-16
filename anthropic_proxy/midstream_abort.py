"""
Mid-stream abort handling for simulating real API connection drops.

When upstream API errors occur mid-stream, the real API behavior is to
drop the TCP connection abruptly. This triggers client-side retry logic.
"""


class MidStreamAbortError(Exception):
    """
    Exception raised to abort a streaming response mid-stream.

    This simulates the real API behavior of dropping connections on errors
    during streaming (e.g., 502, 503 errors). When caught by the middleware,
    the connection is closed cleanly without logging a scary traceback.

    Usage:
        # In streaming generators, when upstream disconnects:
        raise MidStreamAbortError("upstream 502")

    The middleware will catch this and log a clean message:
        [mid-stream abort] upstream 502

    This causes the client connection to drop, triggering agent retry logic.
    """

    pass
