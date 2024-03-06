import asyncio
import os
import typing as t

import numpy as np
import pydantic as pyd
import requests
from loguru import logger

import shiva as shv

HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", 9999)
CAMERA = os.getenv("CAMERA", "camera")


class BoundingBox2D(pyd.BaseModel):
    x: float
    y: float
    w: float = -1.0
    h: float = -1.0
    angle: float = 0.0


class Detection(pyd.BaseModel):
    label: int = -1
    score: float = 0.0
    bbox_2d: t.Optional[BoundingBox2D] = None
    polygon_2d: t.Optional[list] = None
    pose_3d: t.Optional[list] = None
    size_3d: t.Optional[list] = None
    metadata: dict = pyd.Field(default_factory=dict)


class Inference(pyd.BaseModel):
    detections: list[Detection] = pyd.Field(default_factory=list)
    metadata: dict = pyd.Field(default_factory=dict)


async def endpoint_info(message: shv.ShivaMessage) -> shv.ShivaMessage:
    # searching for available vision pipelines
    vision_pipelines = requests.get(
        f"http://{HOST}:{PORT}/vision_pipelines", timeout=10
    )
    # check if the request was successful
    if vision_pipelines.status_code != 200:
        msg = "Cannot get vision pipelines"
        raise Exception(msg)
    vision_pipelines = vision_pipelines.json()
    # check if there is at least one camera
    if len(vision_pipelines) == 0:
        msg = "No vision pipelines available"
        raise Exception(msg)
    metadata = {
        "name": "Rest/Shiva Connector",
        "version": "0.0.1",
        "vision_pipelines": vision_pipelines,
    }
    return shv.ShivaMessage(metadata=metadata, tensors=[])


async def endpoint_inference(message: shv.ShivaMessage) -> shv.ShivaMessage:
    output = requests.get(f"http://{HOST}:{PORT}/inference/{CAMERA}", timeout=10)

    inference = Inference.parse_obj(output.json())
    n_detections = len(inference.detections)
    # label, score, bbox_2d, pose_3d, size_3d
    columns = 1 + 1 + 5 + 16 + 3
    # -1 if not set
    data_array = np.ones((n_detections, columns), dtype=np.float32) * -1

    for i, det in enumerate(inference.detections):
        data_array[i, 0] = det.label
        data_array[i, 1] = det.score

        if det.bbox_2d:
            data_array[i, 2:7] = np.array(
                [
                    det.bbox_2d.x,
                    det.bbox_2d.y,
                    det.bbox_2d.w,
                    det.bbox_2d.h,
                    det.bbox_2d.angle,
                ]
            )

        if det.pose_3d:
            data_array[i, 7 : 7 + 16] = np.array(det.pose_3d).flatten()

        if det.size_3d:
            data_array[i, 7 + 16 : 7 + 16 + 3] = np.array(det.size_3d)

    return shv.ShivaMessage(metadata={}, tensors=[data_array], namespace="inference")


async def endpoint_changeformat(message: shv.ShivaMessage) -> shv.ShivaMessage:
    # setting the preferred status
    status_id = message.metadata["id"]
    response = requests.post(
        f"http://{HOST}:{PORT}/settings/preferred_status/activate/{status_id}",
        timeout=10,
    )
    # check if the request was successful
    if response.status_code != 200:
        msg = "Something went wrong during format change"
        raise Exception(msg)
    return shv.ShivaMessage(metadata={}, tensors=[])


ENDPOINTS_MAP = {
    "info": endpoint_info,
    "inference": endpoint_inference,
    "change-format": endpoint_changeformat,
}


async def manage_message_async(message: shv.ShivaMessage) -> shv.ShivaMessage:
    namespace = message.namespace
    if namespace in ENDPOINTS_MAP:
        return await ENDPOINTS_MAP[namespace](message)
    return shv.ShivaMessage(metadata={}, tensors=[], namespace="error")


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
    logger.info(f"Starting Shiva Server on {HOST}:{PORT} with {CAMERA}")
    asyncio.run(main_async())
