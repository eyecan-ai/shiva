import asyncio
import contextlib
import struct
import threading as th
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pydantic as pyd
import pytest

from shiva import (
    DataPackaging,
    GlobalHeader,
    ShivaBridge,
    ShivaClientAsync,
    ShivaMessage,
    ShivaServer,
    ShivaServerAsync,
    TensorDataTypes,
    TensorHeader,
)

MESSAGES_TO_TEST = [
    # Good simple messages
    (
        {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34], "d": None},
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
    # create pytest with two parameterized arguments, the first is a dict and
    # the second is a list of numpy arrays
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

    # test if a message containing tensors with type id 2 (np.float64, removed as it was
    # overwritten by np.double), can be still correctly parsed
    def test_shiva_message_float64(self):
        message = ShivaMessage(
            metadata={"name": "float64test", "age": 16, "pi": 3.14, "success": True},
            tensors=[np.random.rand(128, 128, 3).astype(np.float64)],
            namespace="float64test",
        )

        def custom_flush(message: ShivaMessage) -> bytes:
            buffer = []
            buffer.append(message.global_header().pack())

            for tensor_header, tensor_shape, tensor_data in zip(
                message.tensors_headers(),
                message.tensors_shapes(),
                message.tensors_data(),
            ):
                # force type id to 2, i.e. the value originally assigned to np.float64
                buffer.append(
                    struct.pack(
                        tensor_header.pack_format(),
                        tensor_header.tensor_rank,
                        2,
                    )
                )
                buffer.append(DataPackaging.pack_ints(tensor_shape))
                buffer.append(tensor_data)

            buffer.append(message.metadata_data())
            buffer.append(message.namespace_data())

            # buffer is a list of bytes, transform it into a single bytes object
            return b"".join(buffer)

        buffer = custom_flush(message)
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

    @pytest.mark.parametrize("metadata, tensors, namespace, errors", MESSAGES_TO_TEST)
    @pytest.mark.parametrize("exception, timeout", [[False, 0], [False, 1], [True, 1]])
    @pytest.mark.asyncio
    async def test_server_exception_sync(
        self, metadata, tensors, namespace, errors, exception, timeout
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
            return ShivaMessage()

        def manage_message_with_errors(message: ShivaMessage) -> ShivaMessage:
            raise RuntimeError()

        message_callback = manage_message_with_errors if exception else manage_message

        server = ShivaServer(
            on_new_message_callback=message_callback,
            on_new_connection=lambda x: print("new connection"),
            on_connection_lost=lambda x: print("connectionlost"),
        )

        server.wait_for_connections(forever=False)
        client = await ShivaClientAsync.create_and_connect()

        if exception:
            with pytest.raises(asyncio.TimeoutError):
                _ = await client.send_message(message, timeout=timeout)
        else:
            _ = await client.send_message(message, timeout=timeout)

        await client.disconnect()
        server.close()

    @pytest.mark.parametrize("metadata, tensors, namespace, errors", MESSAGES_TO_TEST)
    @pytest.mark.parametrize("exception, timeout", [[False, 0], [False, 1], [True, 1]])
    @pytest.mark.asyncio
    async def test_server_exception_async(
        self, metadata, tensors, namespace, errors, exception, timeout
    ):
        if errors is not None:
            with pytest.raises(errors):
                message = ShivaMessage(
                    metadata=metadata, tensors=tensors, namespace=namespace
                )
            return
        else:
            message = ShivaMessage(
                metadata=metadata, tensors=tensors, namespace=namespace
            )

        async def manage_message(message: ShivaMessage) -> ShivaMessage:
            return ShivaMessage()

        async def manage_message_with_errors(message: ShivaMessage) -> ShivaMessage:
            raise RuntimeError()

        message_callback = manage_message_with_errors if exception else manage_message

        server = ShivaServerAsync(
            on_new_message_callback=message_callback,
            on_new_connection=lambda x: print("new connection"),
            on_connection_lost=lambda x: print("connectionlost"),
        )

        await server.wait_for_connections(forever=False)
        client = await ShivaClientAsync.create_and_connect()

        if exception:
            response = await client.send_message(message, timeout=timeout)
            assert response.namespace == "error"
        else:
            response = await client.send_message(message, timeout=timeout)
            assert response.namespace != "error"

        await client.disconnect()
        await server.close()


class TestShivaBridge:
    class Person(ShivaBridge):
        name: str
        age: int
        height: float
        married: bool
        scores: np.ndarray
        children: list
        var: list
        hair: Optional[str]

    def test_obj2msg_msg2obj(self):

        person = TestShivaBridge.Person(
            name="John",
            age=25,
            height=1.75,
            married=False,
            scores=np.random.rand(4, 16, 2, 3),
            children=[
                {
                    "name": "Alice",
                    "pics": [np.random.rand(10, 10, 3) for _ in range(10)],
                },
                {
                    "name": "Bob",
                    "pics": [np.random.rand(12, 11, 3) for _ in range(10)],
                },
            ],
            var=[
                1,
                "two",
                3.0,
                np.array(4.0),
                [
                    np.random.rand(16, 17, 18),
                    {
                        "info": "nothing",
                        "data": np.random.rand(5, 6, 7, 8, 9),
                    },
                ],
            ],
            hair=None,
        )

        expected_metadata = person.dict()
        expected_metadata["scores"] = f"{ShivaBridge.TENSOR}0"
        expected_metadata["children"][0]["pics"] = []
        for i in range(1, 10 + 1):
            expected_metadata["children"][0]["pics"].append(f"{ShivaBridge.TENSOR}{i}")
        expected_metadata["children"][1]["pics"] = []
        for i in range(10 + 1, 20 + 1):
            expected_metadata["children"][1]["pics"].append(f"{ShivaBridge.TENSOR}{i}")
        expected_metadata["var"][3] = f"{ShivaBridge.TENSOR}{21}"
        expected_metadata["var"][4][0] = f"{ShivaBridge.TENSOR}{22}"
        expected_metadata["var"][4][1]["data"] = f"{ShivaBridge.TENSOR}{23}"
        expected_metadata["hair"] = "null"

        expected_tensors = [person.scores]
        expected_tensors.extend(person.children[0]["pics"])
        expected_tensors.extend(person.children[1]["pics"])
        expected_tensors.append(person.var[3])
        expected_tensors.append(person.var[4][0])
        expected_tensors.append(person.var[4][1]["data"])

        expected_msg = ShivaMessage(
            metadata=expected_metadata,
            tensors=expected_tensors,
            namespace="Person",
        )

        msg = person.to_shiva_message(namespace="Person")

        assert msg == expected_msg

        rebuilt_person = TestShivaBridge.Person.from_shiva_message(msg)

        assert person.name == rebuilt_person.name
        assert person.age == rebuilt_person.age
        assert person.height == rebuilt_person.height
        assert person.married == rebuilt_person.married
        assert np.all(person.scores == rebuilt_person.scores)

        assert len(person.children) == len(rebuilt_person.children)
        for c1, c2 in zip(person.children, rebuilt_person.children):
            assert c1["name"] == c2["name"]
            assert len(c1["pics"]) == len(c2["pics"])
            for p1, p2 in zip(c1["pics"], c2["pics"]):
                assert np.all(p1 == p2)

        v1, v2 = person.var, rebuilt_person.var
        assert v1[0] == v2[0]
        assert v1[1] == v2[1]
        assert v1[2] == v2[2]
        assert np.all(v1[3] == v2[3])
        assert np.all(v1[4][0] == v2[4][0])
        assert v1[4][1]["info"] == v2[4][1]["info"]
        assert np.all(v1[4][1]["data"] == v2[4][1]["data"])

    def test_unsupported_type(self):
        class MyObject(ShivaBridge):
            kind: str
            unknown: Any

        wrong_objs = [
            MyObject(kind="a", unknown=lambda x: x),
            MyObject(kind="b", unknown=th.Lock()),
            MyObject(kind="c", unknown=datetime(2021, 1, 1, tzinfo=timezone.utc)),
        ]

        for obj in wrong_objs:
            error_msg = f"ShivaBridge unsupported type {type(obj.unknown)}"
            with pytest.raises(ValueError, match=error_msg):
                obj.to_shiva_message()

    @pytest.mark.asyncio
    async def test_native_byteorder(self) -> None:

        def manage_message(message: ShivaMessage) -> ShivaMessage:
            return message

        server = ShivaServer(
            on_new_message_callback=manage_message,
            on_new_connection=lambda x: print("new connection"),
            on_connection_lost=lambda x: print("connectionlost"),
        )

        server.wait_for_connections(forever=False)

        client = await ShivaClientAsync.create_and_connect()

        person = TestShivaBridge.Person(
            name="Alice",
            age=32,
            height=1.58,
            married=True,
            scores=np.random.rand(2, 3, 8, 9),
            children=[
                {
                    "name": "Bob",
                    "pics": [np.random.rand(16, 16, 3) for _ in range(3)],
                },
            ],
            var=[],
        )

        response_message = await client.send_message(person.to_shiva_message())
        rebuilt_person = TestShivaBridge.Person.from_shiva_message(response_message)
        rebuilt_tensors = [rebuilt_person.scores, *rebuilt_person.children[0]["pics"]]

        # check that all rebuilt tensors are returned with native byteorder
        for tensor in rebuilt_tensors:
            assert tensor.dtype.byteorder == "="

        await client.disconnect()
        server.close()
