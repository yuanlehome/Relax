# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Forward host clients to an authenticated Gym on an OCI-capable worker.

The byte stream is opaque: management authentication and rollout capabilities
remain enforced by Gym. Connections have no read deadline, allowing resident
model requests to survive training pauses.
"""

from __future__ import annotations

import argparse
import asyncio
import signal

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


async def forward(host: str, port: int, upstream_host: str, upstream_port: int) -> None:
    connections: set[asyncio.Task[None]] = set()

    async def copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        upstream_writer = None
        pumps = []
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(upstream_host, upstream_port), timeout=10
            )
            pumps = [
                asyncio.create_task(copy(reader, upstream_writer)),
                asyncio.create_task(copy(upstream_reader, writer)),
            ]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        except (OSError, TimeoutError):
            logger.warning("Gym forwarding connection failed")
        finally:
            for pump in pumps:
                pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for stream in (writer, upstream_writer):
                if stream is not None:
                    stream.close()
                    try:
                        await stream.wait_closed()
                    except OSError:
                        pass
            connections.discard(task)

    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    server = await asyncio.start_server(handle, host, port)
    logger.info("Gym forwarding ready at %s:%s", host, port)
    try:
        async with server:
            await stopped.wait()
    finally:
        for task in tuple(connections):
            task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(forward(args.host, args.port, args.upstream_host, args.upstream_port))


if __name__ == "__main__":
    main()
