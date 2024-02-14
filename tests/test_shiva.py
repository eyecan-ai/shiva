import threading as th
from datetime import datetime, timezone
from typing import Any

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


class TestShivaBridge:
    def test_obj2msg_msg2obj(self):

        class Person(ShivaBridge):
            name: str
            age: int
            height: float
            married: bool
            scores: np.ndarray
            children: list
            var: list

        person = Person(
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

        rebuilt_person = Person.from_shiva_message(msg)

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
