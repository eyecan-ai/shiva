import asyncio
from shiva import ShivaClientAsync, ShivaMessage
import shiva
import numpy as np
import time


async def tcp_echo_client():
    message = ShivaMessage(
        metadata={"time": time.time()},
        tensors=[
            np.random.randint(0, 255, (1920, 180, 3)).astype(np.uint8),
            np.random.randint(0, 255, (1920, 180, 3)).astype(np.uint8),
            np.random.randint(0, 255, (1920, 180, 3)).astype(np.uint8),
        ],
        namespace="timings",
    )

    client = await ShivaClientAsync.create_and_connect("localhost", 8888)

    from aiolimiter import AsyncLimiter

    rate_limit = AsyncLimiter(60, 1)

    while True:
        async with rate_limit:
            message.metadata["time"] = time.time()
            received_message = await client.send_message(message)

            time_difference = time.time() - received_message.metadata["time"]

            # print roundtrip time in milliseconds
            print(f"Roundtrip time: {time_difference * 1000:.2f} ms")

            hz = 1 / time_difference
            print(f"Hz: {hz:.2f}")


asyncio.run(tcp_echo_client())
