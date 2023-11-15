import asyncio
import shiva as shv
import rich


async def example_client():
    # Create an async Shiva client
    client = await shv.ShivaClientAsync.create_and_connect("10.1.70.50")

    # Send message and grab response
    response = await client.send_message(shv.ShivaMessage(namespace='info'))
    rich.print("INFO", response.metadata)
    response = await client.send_message(shv.ShivaMessage(namespace='inference'))
    rich.print("INFERENCE", response)
    response = await client.send_message(shv.ShivaMessage(namespace='BAD'))
    rich.print("BAD", response)


if __name__ == '__main__':
    asyncio.run(example_client())
