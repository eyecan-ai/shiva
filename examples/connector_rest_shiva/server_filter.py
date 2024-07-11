import asyncio
import os

import cv2 as cv
import numpy as np
import requests
from loguru import logger
from server import Inference, endpoint_changeformat, endpoint_info

import shiva as shv

HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", "9999")
CAMERA = os.getenv("CAMERA", "camera")
HEIGHT = int(os.getenv("HEIGHT", "1080"))
WIDTH = int(os.getenv("WIDTH", "1920"))
IOU_FILTER_THRESHOLD = float(os.getenv("IOU_FILTER_THRESHOLD", "0.1"))
FILTER_NON_PICKABLE = os.getenv("FILTER_NON_PICKABLE", "true").lower() == "true"
EXTRA_CLASS_NAME = os.getenv("EXTRA_CLASS_NAME", "pickable")


def check_pickable(
    bboxes: np.ndarray,
    canvas_hw: tuple[int, int],
    th: float,
    ss: int,
) -> np.ndarray:
    height, width = canvas_hw
    num = len(bboxes)
    """ Check if a bounding box is pickable based on the IOU of the bounding boxes in the canvas

    Returns:
        array of bools indicating if the bounding box is pickable or not based on IOU threshold
    """

    canvas_stack = np.zeros((num, height // ss, width // ss), dtype=np.float32)

    bboxes_ss = bboxes[:, :4] / ss
    for i, bbox in enumerate(bboxes_ss):
        x, y, w, h = bbox
        angle = int(bboxes[i, 4])
        pts = cv.boxPoints(((x, y), (w, h), angle)).astype(np.int32)  # type: ignore
        cv.fillPoly(canvas_stack[i], [pts], 1)  # type: ignore

    canvas_sum = np.sum(canvas_stack, axis=0)
    canvas = np.clip(canvas_sum, 0, 2)
    canvas_eq1 = bboxes_ss[:, 2] * bboxes_ss[:, 3]
    canvas_eq2 = (canvas * canvas_stack - 1).clip(0, None).sum(axis=(1, 2))
    return canvas_eq2 / canvas_eq1 < th


async def endpoint_inference(message: shv.ShivaMessage) -> shv.ShivaMessage:
    output = requests.get(f"http://{HOST}:{PORT}/inference/{CAMERA}", timeout=10)

    if output.status_code != 200:
        logger.error(f"Error in inference request: {output.status_code}")
        return shv.ShivaMessage(metadata={}, tensors=[], namespace="inference")

    inference = Inference.parse_obj(output.json())

    n_detections = len(inference.detections)
    # label, score, bbox_2d, pose_3d, size_3d, pickable_flag
    columns = 1 + 1 + 5 + 16 + 3 + 1
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

        if EXTRA_CLASS_NAME in det.metadata:
            data_array[i, -1] = det.metadata.get(EXTRA_CLASS_NAME)

    # filter out detections with underthreshold IOU overlap
    if IOU_FILTER_THRESHOLD < 1:
        valid_detections = check_pickable(
            data_array[:, 2:7], (HEIGHT, WIDTH), IOU_FILTER_THRESHOLD, ss=2
        )
        logger.info(
            f"detections after IOU {IOU_FILTER_THRESHOLD} thresholding: {valid_detections.sum()} / {n_detections}"
        )
        data_array = data_array[valid_detections]

    # filter out non pickable detections
    if FILTER_NON_PICKABLE:
        pickable_flag = data_array[:, -1] == 1
        data_array = data_array[pickable_flag]
        logger.info(
            f"detections after non-pickable filter: { len(data_array) } / {valid_detections.sum()}"
        )

    return shv.ShivaMessage(metadata={}, tensors=[data_array], namespace="inference")


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
