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
    ShivaServerAsync,
    ShivaClientAsync,
)
import pydantic as pyd

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

        global_header_chunk = buffer[: GlobalHeader.pack_size()]
        buffer = buffer[GlobalHeader.pack_size() :]
        global_header = GlobalHeader.unpack(global_header_chunk)

        metadata_size = global_header.metadata_size
        n_tensors = global_header.n_tensors

        # receive the tensors headers
        tensors_headers: List[TensorHeader] = []
        for _ in range(n_tensors):
            # receive a single tensor header
            tensor_header_chunk = buffer[: TensorHeader.pack_size()]
            buffer = buffer[TensorHeader.pack_size() :]

            tensor_header = TensorHeader.unpack(tensor_header_chunk)
            tensors_headers.append(tensor_header)

        # receive the tensors shapes
        tensor_shapes: List[List[int]] = []
        for idx, _ in enumerate(range(n_tensors)):
            # the size of the shape is 4 bytes per dimension
            shape_size = 4 * tensors_headers[idx].tensor_rank

            # receive the shape
            shape_data = buffer[:shape_size]
            buffer = buffer[shape_size:]
            shape = DataPackaging.unpack_ints(shape_data)
            tensor_shapes.append(shape)

        # receive the tensors data
        tensors: List[np.ndarray] = []

        for idx, _ in enumerate(range(n_tensors)):
            # the size of the data is the product of the shape elements times the size
            # of the tensor data type (byte-size)
            bytes_per_element = np.dtype(tensors_headers[idx].tensor_dtype).itemsize
            expected_data = np.prod(tensor_shapes[idx]) * bytes_per_element

            # receive the data

            data = buffer[:expected_data]
            buffer = buffer[expected_data:]

            # convert the data into a numpy array
            t = np.frombuffer(
                data,
                dtype=tensors_headers[idx].tensor_dtype,
            ).reshape(tensor_shapes[idx])

            tensors.append(t)

        # receive the metadata if any
        metadata = {}
        if metadata_size > 0:
            data = buffer[:metadata_size]
            buffer = buffer[metadata_size:]
            metadata = json.loads(data.decode("utf-8"))

        namespace = ""
        if global_header.tail_string_size > 0:
            data = buffer[: global_header.tail_string_size]
            namespace = data.decode("utf-8")

        rebuilt_message = ShivaMessage(
            metadata=metadata,
            tensors=tensors,
            namespace=namespace,
        )

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
