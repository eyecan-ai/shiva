import asyncio
import time
from typing import Dict

import numpy as np

from shiva import ShivaMessage, ShivaServerAsync, ShivaServer
from loguru import logger
from queue import Queue

shared_queue_map: Dict[tuple, asyncio.Queue] = {}
results_queue = asyncio.Queue(1)


def manage_message_async(message: ShivaMessage) -> ShivaMessage:
    # if message.sender not in shared_queue_map:
    #     shared_queue_map[message.sender] = asyncio.Queue(1)
    print("METADATA", message.metadata)
    print("NAMESPACE", message.namespace)
    new_tensors = []
    for tensor in message.tensors:
        print(tensor.shape, tensor.dtype)
        new_tensor = tensor + 0.01
        new_tensors.append(new_tensor)

    # q = shared_queue_map[message.sender]

    # try:
    #     q.put_nowait(message.tensors[0].copy())
    # except:
    #     pass

    return ShivaMessage(metadata=message.metadata, tensors=new_tensors)


async def main_async():
    server = ShivaServer(
        on_new_message_callback=manage_message_async,
        on_new_connection=lambda x: logger.info(f"New connection -> {x}"),
        on_connection_lost=lambda x: logger.error(f"Connection lost -> {x}"),
    )
    server.wait_for_connections(forever=True)
    # server = ShivaServerAsync(
    #     on_new_message_callback=manage_message_async,
    #     on_new_connection=lambda x: logger.info(f"New connection -> {x}"),
    #     on_connection_lost=lambda x: logger.error(f"Connection lost -> {x}"),
    # )
    # await server.wait_for_connections(forever=True)

    # while True:
    #     await asyncio.sleep(0.0001)

    #     t1 = time.perf_counter()
    #     found = []
    #     for sender, shared_queue in shared_queue_map.items():
    #         if not shared_queue.empty():
    #             found.append(sender)

    #     if len(found) != 4:  # Change this based on number of expected clients
    #         continue
    #     else:
    #         found = []
    #         for sender, shared_queue in shared_queue_map.items():
    #             try:
    #                 found.append(shared_queue.get_nowait())
    #             except Exception as e:
    #                 print(e)

    #     if len(found) == 0:
    #         print("No data")
    #     else:
    #         mixup = np.array(found)
    #         uniques = np.unique(mixup)

    #         try:
    #             results_queue.put(uniques)
    #         except:
    #             pass

    #         try:
    #             print("Results", len(results_queue.get_nowait()))
    #         except Exception as e:
    #             print(e)
    #             pass

    #         t2 = time.perf_counter()
    #         hz = 1 / (t2 - t1)
    #         print(f"Hz: {hz:.2f}")


if __name__ == "__main__":
    asyncio.run(main_async())
