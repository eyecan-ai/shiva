import asyncio
import shiva as shv
import numpy as np
import time
import rich


async def example_client():
    # Create a Shiva Message, with metadata, namespace and tensors
    message = shv.ShivaMessage(
        metadata={
            # This is a special key used to name the tensors in order, this is useful
            # for business logic only, it does not change the protocol behaviour
            shv.ShivaConstants.TENSORS_KEY: ['left', 'right', 'counter'],
            "counter": 0,
        },
        tensors=[
            np.random.randint(0, 255, (1920, 1080, 3)).astype(np.uint8),  # left
            np.random.randint(0, 255, (1920, 1080, 3)).astype(np.uint8),  # right
            np.array([0]).astype(np.int64),  # corner
        ],
        namespace="inference",
    )

    # Create an async Shiva client
    client = await shv.ShivaClientAsync.create_and_connect("localhost")

    while True:
        # Send message and grab response
        t1 = time.perf_counter()
        message = await client.send_message(message)
        t2 = time.perf_counter()

        # Computer RoundTrip time/fps
        time_difference = t2 - t1
        hz = 1 / time_difference
        print(f"Roundtrip time: {time_difference * 1000:.2f} ms", f"Hz: {hz:.2f}")

        # Print output, if example server running it should increment 'counter' value
        # in metadata and value of 'counter' tensor
        rich.print(message.metadata)
        if len(message.tensors) >= 3:
            rich.print(message.tensors[2], message.tensors[2].dtype)


if __name__ == '__main__':
    asyncio.run(example_client())
