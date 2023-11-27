import json
from typing import List
import pytest
import numpy as np
from shiva import (
    ShivaMessage,
    GlobalHeader,
    TensorHeader,
    DataPackaging,
    TensorDataTypes,
    ShivaServer,
    ShivaServerAsync,
    ShivaClientAsync,
)
import pydantic as pyd
import threading as th
import time

MESSAGES_TO_TEST = [
    # Good simple messages
    (
        {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
        [
            np.zeros((128, 128, 3)).astype(x)
            for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
        ],
        "namespace",
        None,
    ),
    # (
    #     {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
    #     [
    #         np.zeros((1, 2, 3, 4, 5)).astype(x)
    #         for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
    #     ],
    #     "namespace",
    #     None,
    # ),
    # (
    #     {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
    #     [np.zeros((100,)).astype(x) for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()],
    #     "namespace",
    #     None,
    # ),
    # (
    #     {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
    #     [np.zeros((0,)).astype(x) for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()],
    #     "namespace",
    #     None,
    # ),
    # (
    #     {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
    #     [np.zeros((0,)).astype(x) for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()],
    #     "",
    #     None,
    # ),
    (
        {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
        [np.zeros((0,)).astype(x) for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()],
        "x",
        None,
    ),
    # # Bad Message with wrong tensor type
    (
        {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
        None,
        "namespace",
        pyd.ValidationError,
    ),
    # # Bad Message with wrong metadata and tensor type
    (
        None,
        None,
        "namespace",
        pyd.ValidationError,
    ),
    # # Bad Message with wrong metadata type
    (
        None,
        [
            np.zeros((128, 128, 3)).astype(x)
            for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
        ],
        "namespace",
        pyd.ValidationError,
    ),
]


class TestShivaMessage:
    # create pytest with two parameterized arguments , the first is a dict and the second is
    # a list of numpy arrays
    @pytest.mark.parametrize("metadata, tensors, namespace, errors", MESSAGES_TO_TEST)
    def test_shiva_message(self, metadata, tensors, namespace, errors):
        if errors is not None:
            with pytest.raises(errors):
                message = ShivaMessage(
                    metadata=metadata,
                    tensors=tensors,
                    namespace=namespace,
                )
            return
        else:
            message = ShivaMessage(
                metadata=metadata,
                tensors=tensors,
                namespace=namespace,
            )

        buffer = message.flush()
        rebuilt_message = ShivaMessage.parse(buffer)
        assert message == rebuilt_message

    @pytest.mark.parametrize("metadata, tensors, namespace, errors", MESSAGES_TO_TEST)
    @pytest.mark.asyncio
    async def test_shiva_message2(self, metadata, tensors, namespace, errors):
        if errors is not None:
            with pytest.raises(errors):
                message = ShivaMessage(
                    metadata=metadata,
                    tensors=tensors,
                    namespace=namespace,
                )
            return
        else:
            message = ShivaMessage(
                metadata=metadata,
                tensors=tensors,
                namespace=namespace,
            )

        async def manage_message(message: ShivaMessage) -> ShivaMessage:
            return message

        server = ShivaServerAsync(
            on_new_message_callback=manage_message,
            on_new_connection=lambda x: print("new connection"),
            on_connection_lost=lambda x: print("connectionlost"),
        )

        # asyncio.set_event_loop(loop)

        await server.wait_for_connections(forever=False)

        client = await ShivaClientAsync.create_and_connect()
        client2 = await ShivaClientAsync.create_and_connect()

        response_message = await client.send_message(message)

        assert response_message == message

        await server.close()

    @pytest.mark.parametrize("metadata, tensors, namespace, errors", MESSAGES_TO_TEST)
    @pytest.mark.asyncio
    async def test_shiva_message_server_sync(
        self, metadata, tensors, namespace, errors
    ):
        if errors is not None:
            with pytest.raises(errors):
                message = ShivaMessage(
                    metadata=metadata,
                    tensors=tensors,
                    namespace=namespace,
                )
            return
        else:
            message = ShivaMessage(
                metadata=metadata,
                tensors=tensors,
                namespace=namespace,
            )

        def manage_message(message: ShivaMessage) -> ShivaMessage:
            return message

        server = ShivaServer(
            on_new_message_callback=manage_message,
            on_new_connection=lambda x: print("new connection"),
            on_connection_lost=lambda x: print("connectionlost"),
        )

        server.wait_for_connections(forever=False)

        client = await ShivaClientAsync.create_and_connect()
        client2 = await ShivaClientAsync.create_and_connect()

        response_message = await client.send_message(message)

        assert response_message == message
        await client.disconnect()
        await client2.disconnect()
        server.close()
