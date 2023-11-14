import asyncio
from typing import Dict
import shiva as shv
from loguru import logger
import numpy as np


async def endpoint_info(message: shv.ShivaMessage) -> shv.ShivaMessage:
    metadata = {
        'name': 'Shiva Inference Server',
        'version': '0.0.1',
    }
    return shv.ShivaMessage(metadata=metadata, tensors=[])


async def endpoint_inference(message: shv.ShivaMessage) -> shv.ShivaMessage:
    N = 16
    M = 8
    data = np.arange(N * M).reshape(N, M).astype(np.float32)
    return shv.ShivaMessage(metadata={}, tensors=[data], namespace='inference')


ENDPOINTS_MAP = {
    'info': endpoint_info,
    'inference': endpoint_inference,
}


async def manage_message_async(message: shv.ShivaMessage) -> shv.ShivaMessage:
    namespace = message.namespace
    if namespace in ENDPOINTS_MAP:
        return await ENDPOINTS_MAP[namespace](message)
    return shv.ShivaMessage(metadata={}, tensors=[], namespace='error')


async def main_async():
    # Creates a Shive Server in Async mode
    server = shv.ShivaServerAsync(
        # this is the callback managing all incoming messages
        on_new_message_callback=manage_message_async,
        # this callback is useful to manage onConnect events
        on_new_connection=lambda x: logger.info(f"New connection -> {x}"),
        # this callback is useful to manage onDisconnect events
        on_connection_lost=lambda x: logger.error(f"Connection lost -> {x}"),
    )
    await server.wait_for_connections(forever=True)


if __name__ == "__main__":
    asyncio.run(main_async())
