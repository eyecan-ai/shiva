import asyncio
from typing import Dict
import shiva as shv
from loguru import logger

shared_queue_map: Dict[tuple, asyncio.Queue] = {}
results_queue = asyncio.Queue(1)


async def manage_message_async(message: shv.ShivaMessage) -> shv.ShivaMessage:
    """This manage function should inspect the tensors and add +1 to the tensor
    called 'counter'. The name of the tensor is listed in the special key in metadata.
    Moreover, it adds +1 to the 'counter' value in metadata if any
    """

    # Retrieve tensors name, if provided it should be in the special key of metadata
    tensors_names = message.metadata.get(shv.ShivaConstants.TENSORS_KEY, [])
    current_metadata = message.metadata.copy()

    # if 'counter' , add +1
    if 'counter' in current_metadata:
        current_metadata['counter'] += 1

    # browse tensors and add +1 to the one called 'counter'
    new_tensors = []
    for idx, tensor in enumerate(message.tensors):
        name = tensors_names[idx] if idx < len(tensors_names) else ''
        if 'counter' in name:
            new_tensor = tensor.copy()
            new_tensor += 1
            new_tensors.append(new_tensor)
        else:
            new_tensors.append(tensor)

    return shv.ShivaMessage(metadata=current_metadata, tensors=new_tensors)


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
