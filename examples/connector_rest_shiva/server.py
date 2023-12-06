import asyncio
import typing as t

import numpy as np
import pydantic as pyd
import requests
from loguru import logger

import shiva as shv


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
    metadata = {
        'name': 'Shiva Inference Server',
        'version': '0.0.1',
    }
    return shv.ShivaMessage(metadata=metadata, tensors=[])


async def endpoint_inference(message: shv.ShivaMessage) -> shv.ShivaMessage:
    # searching for available vision pipelines
    vision_pipelines = requests.get(
        "http://localhost:9999/vision_pipelines", timeout=10
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
    # use first camera available
    camera_name = vision_pipelines[0]
    output = requests.get(f"http://localhost:9999/inference/{camera_name}", timeout=10)

    inference = Inference.parse_obj(output.json())
    n_detections = len(inference.detections)
    # label, score, bbox_2d, pose_3d.
    columns = 1 + 1 + 5 + 16
    # -1 if not set
    data_array = np.ones((n_detections, columns), dtype=np.float32) * -1

    def pose_3d_2_bbox_2d(
        pose_3d: t.Union[np.ndarray, list],
        size_3d: t.Optional[t.Union[np.array, list]] = None,
    ) -> BoundingBox2D:
        if isinstance(pose_3d, list):
            pose_3d = np.array(pose_3d)
        t = pose_3d[:3, 3]
        w = -1
        h = -1
        if size_3d:
            if isinstance(size_3d, list):
                size_3d = np.array(size_3d)
            w, h, _ = size_3d
        angle = 0
        # rotate the bbox_3d to get the angle
        c_rot_w = pose_3d[:3, :3]
        angle = np.rad2deg(np.arctan2(c_rot_w[1, 0], c_rot_w[0, 0]))
        return BoundingBox2D(
            x=t[0],
            y=t[1],
            w=w,
            h=h,
            angle=angle,
        )

    for i, det in enumerate(inference.detections):
        data_array[i, 0] = det.label
        data_array[i, 1] = det.score
        if det.pose_3d:
            data_array[i, 7:] = np.array(det.pose_3d).flatten()
        else:
            logger.warning("pose_3d is None")
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
        else:
            if det.pose_3d:
                bbox_2d = pose_3d_2_bbox_2d(pose_3d=det.pose_3d, size_3d=det.size_3d)
                data_array[i, 2:7] = np.array(
                    [bbox_2d.x, bbox_2d.y, bbox_2d.w, bbox_2d.h, bbox_2d.angle]
                )
            logger.warning("bbox_2d is None")
    return shv.ShivaMessage(metadata={}, tensors=[data_array], namespace='inference')


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
