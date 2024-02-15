import argparse
import asyncio
import os

import rich

import shiva as shv

STATUS_ID = os.getenv("STATUS_ID", "1234")


async def example_client(args: argparse.Namespace):
    # Create an async Shiva client
    client = await shv.ShivaClientAsync.create_and_connect("localhost")

    # Send message and grab response
    response = await client.send_message(shv.ShivaMessage(namespace="info"))
    rich.print("INFO", response.metadata)

    response = await client.send_message(shv.ShivaMessage(namespace="inference"))
    rich.print("INFERENCE", response)

    response = await client.send_message(shv.ShivaMessage(namespace="BAD"))
    rich.print("BAD", response)

    if args.status_id:

        msg = shv.ShivaMessage(
            namespace="change-format",
            metadata={"id": f"{args.status_id}"},
        )

        rich.print(msg.global_header().pack().hex())
        rich.print(msg.metadata_data().hex())
        rich.print(msg.namespace_data().hex())

        response = await client.send_message(msg)
        rich.print("CHANGE FORMAT", response)


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--status_id", type=str)
    args = parser.parse_args()

    asyncio.run(example_client(args))
