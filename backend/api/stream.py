"""Server-Sent Events stream for the Monitor page.

Tails ``runs/queue_events.log`` (structured status/epoch/done/log events) and
``runs/queue_output.log`` (raw training stdout) and emits periodic ``status``
heartbeats. Closes when the client disconnects.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.config import EVENTS_LOG, OUTPUT_LOG
from .. import process

router = APIRouter()

HEARTBEAT_SECONDS = 2
_TAIL_BYTES = 16384  # initial context window per log on connect


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _initial_pos(path, tail_bytes: int) -> int:
    """Return a starting byte offset ~tail_bytes from EOF (line-aligned)."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    return max(0, size - tail_bytes)


def _read_new(path, pos: int):
    """Yield (lines, new_pos) for content appended after ``pos``."""
    try:
        with open(path, "r", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            new_pos = f.tell()
    except OSError:
        return [], pos
    if not chunk:
        return [], new_pos
    lines = chunk.splitlines()
    # If the chunk didn't end on a newline, keep the partial line for next read.
    if not chunk.endswith("\n") and lines:
        new_pos -= len(lines[-1].encode("utf-8", "replace"))
        lines = lines[:-1]
    return lines, new_pos


async def _generate(request: Request, shutdown_event: asyncio.Event):
    yield ": connected\n\n"
    yield _sse("status", process.status())

    ev_pos = _initial_pos(EVENTS_LOG, _TAIL_BYTES)
    out_pos = _initial_pos(OUTPUT_LOG, _TAIL_BYTES)

    while True:
        if await request.is_disconnected():
            break

        ev_lines, ev_pos = _read_new(EVENTS_LOG, ev_pos)
        for line in ev_lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                etype = obj.get("type", "log")
            except json.JSONDecodeError:
                obj, etype = {"message": line}, "log"
            yield _sse(etype, obj)

        out_lines, out_pos = _read_new(OUTPUT_LOG, out_pos)
        for line in out_lines:
            if line.strip():
                yield _sse("log", {"message": line, "source": "stdout"})

        yield _sse("status", process.status())

        # Sleep for HEARTBEAT_SECONDS, but wake immediately on server shutdown
        # so uvicorn doesn't hang waiting for this generator to finish.
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=HEARTBEAT_SECONDS)
            break  # shutdown signaled
        except asyncio.TimeoutError:
            pass  # normal tick


@router.get("/queue/stream")
async def queue_stream(request: Request) -> StreamingResponse:
    shutdown_event: asyncio.Event = request.app.state.shutdown_event
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_generate(request, shutdown_event),
                             media_type="text/event-stream", headers=headers)
