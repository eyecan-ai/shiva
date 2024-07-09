import asyncio
import secrets as sc
import socket
import struct
import threading
import threading as th
import time
import typing as t
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pydantic as pyd
import pytest

from shiva import (
    DataPackaging,
    GlobalHeader,
    ShivaBridge,
    ShivaClientAsync,
    ShivaErrorMessage,
    ShivaMessage,
    ShivaServer,
    ShivaServerAsync,
    TensorDataTypes,
    TensorHeader,
)


# Helper functions
@contextmanager
def dont_raise():
    yield None


async def cb_async(m):
    return m


def cb_sync(m):
    return m


async def cb_async_tout(m, t=2):
    await asyncio.sleep(t)
    return m


def cb_sync_tout(m, t=2):
    time.sleep(t)
    return m


def cb_sync_exc(_):
    raise Exception()


async def cb_async_exc(_):
    raise Exception()


class TestShivaModel:
    TEST_DATA_PACKAGING: t.ClassVar = [
        (b"ciao", dont_raise()),
        (b"miao"[:-2], pytest.raises(ValueError)),  # missing 2 bytes
    ]
    TEST_CRC: t.ClassVar = [
        (-1, pytest.raises(ValueError)),  # change crc1
        (-2, pytest.raises(ValueError)),  # change crc2
        # change random byte, crcs should keep us safe
        (sc.randbelow(len(GlobalHeader.pack_format()) - 1), pytest.raises(ValueError)),
    ]

    @pytest.mark.parametrize("data, expectation", TEST_DATA_PACKAGING)
    def test_data_packaging(self, data, expectation):
        with expectation:
            DataPackaging.unpack_ints(data)

    @pytest.mark.parametrize("crc_index, expectation", TEST_CRC)
    def test_crc(self, crc_index, expectation):
        data = GlobalHeader(metadata_size=1, n_tensors=2).pack()
        elements = list(struct.unpack(GlobalHeader.pack_format(), data))
        elements[crc_index] = elements[crc_index] + 1

        with expectation:
            GlobalHeader.unpack(struct.pack(GlobalHeader.pack_format(), *elements))

    def test_magic_numbers(self, monkeypatch):

        data = GlobalHeader(metadata_size=1, n_tensors=2).pack()
        assert TensorHeader.magic_number() is None
        monkeypatch.setattr(GlobalHeader, "magic_number", classmethod(lambda _: None))
        GlobalHeader.unpack(data)

        monkeypatch.setattr(GlobalHeader, "magic_number", classmethod(lambda _: (2, 1)))

        with pytest.raises(ValueError):
            elements = list(struct.unpack(GlobalHeader.pack_format(), data))
            mn = GlobalHeader.magic_number()
            assert mn is not None
            elements[: len(mn)].sort()
            GlobalHeader.unpack(struct.pack(GlobalHeader.pack_format(), *elements))


class TestShivaMessage:
    TEST_MESSAGES: t.ClassVar = [
        # Good simple messages
        (
            {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34], "d": None},
            [
                np.zeros((128, 128, 3)).astype(x)
                for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
            ],
            "namespace",
            dont_raise(),
        ),
        # Good simple messages with another namespace
        (
            {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
            [
                np.zeros((0,)).astype(x)
                for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
            ],
            "x",
            dont_raise(),
        ),
        # # Bad Message with wrong tensor type
        (
            {"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
            None,
            "namespace",
            pytest.raises(pyd.ValidationError),
        ),
        # # Bad Message with wrong metadata and tensor type
        (
            None,
            None,
            "namespace",
            pytest.raises(pyd.ValidationError),
        ),
        # # Bad Message with wrong metadata type
        (
            None,
            [
                np.zeros((128, 128, 3)).astype(x)
                for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
            ],
            "namespace",
            pytest.raises(pyd.ValidationError),
        ),
        # # Message with empty stuff
        (
            {},
            [],
            "",
            dont_raise(),
        ),
    ]

    # Here we test that the shiva message is correctly built and parsed
    @pytest.mark.parametrize("meta, tensors, namespace, msg_exp", TEST_MESSAGES)
    def test_messages(self, meta, tensors, namespace, msg_exp):
        with msg_exp:
            message = ShivaMessage(metadata=meta, tensors=tensors, namespace=namespace)
            assert message == ShivaMessage.parse(message.flush())

    # Here we test that the shiva message is correctly built and parsed even if
    # the tensor type is no more supported by the DataPackaging class
    # (np.float64 has been replaced by np.double
    def test_float64_message(self):
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

    def test_message_eq(self):
        meta, tensors, namespace = TestShivaMessage.TEST_MESSAGES[0][:3]
        message = ShivaMessage(metadata=meta, tensors=tensors, namespace=namespace)
        # Not equal to other objects
        other = 1
        assert not message == other

        # Not equal to messages with different namespace
        other = ShivaMessage(metadata=meta, tensors=tensors, namespace="other")
        assert not message == other

        # Not equal to messages with different metadata
        other = ShivaMessage(
            metadata={"other": "meta"}, tensors=tensors, namespace=namespace
        )
        assert not message == other

        # Not equal to messages with different tensors length
        other = ShivaMessage(metadata=meta, tensors=tensors[:1], namespace=namespace)
        assert not message == other

        # Not equal to messages with different tensors content
        other_tensors = [
            np.ones((128, 128, 3)).astype(x)
            for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
        ]
        other = ShivaMessage(
            metadata=meta,
            tensors=other_tensors,
            namespace=namespace,
        )
        assert not message == other


class TestShivaServer:

    GOOD_MESSAGE: t.ClassVar = ShivaMessage(
        namespace="namespace",
        metadata={"a": 2, "b": 3.145, "s": "a_String", "l": [1, 2, 34]},
        tensors=[
            np.zeros((128, 128, 3)).astype(x)
            for x in TensorDataTypes.RAW_NUMPY_2_DTYPE.keys()
        ],
    )
    EMPTY_MESSAGE: t.ClassVar = ShivaMessage()

    TEST_BASE: t.ClassVar = [
        (ShivaServer, cb_sync_tout, 1, pytest.raises(asyncio.TimeoutError)),
        (ShivaServerAsync, cb_async_tout, 1, pytest.raises(asyncio.TimeoutError)),
        (ShivaServer, cb_sync, 0, dont_raise()),
        (ShivaServerAsync, cb_async, 0, dont_raise()),
    ]

    TEST_FOREVER: t.ClassVar = [
        (ShivaServer, cb_sync, True, pytest.raises(ConnectionRefusedError)),
        (ShivaServerAsync, cb_async, True, pytest.raises(ConnectionRefusedError)),
        (ShivaServer, cb_sync, False, dont_raise()),
        (ShivaServerAsync, cb_async, False, dont_raise()),
    ]

    TEST_ERROR_LOGGING: t.ClassVar = [
        (ShivaServer, cb_sync_exc, dont_raise(), ShivaErrorMessage(Exception())),
        (ShivaServerAsync, cb_async_exc, dont_raise(), ShivaErrorMessage(Exception())),
    ]

    TEST_CLOSING: t.ClassVar = [
        (ShivaServer, cb_sync, pytest.raises(RuntimeError)),
        (ShivaServerAsync, cb_async, pytest.raises(RuntimeError)),
    ]

    # Here we test:
    # 1. The server is correctly created and the callbacks are correctly set
    # 2. The server can accept multiple clients
    # 3. The server can correctly send and receive messages
    # 4. The client raises an exception if the server is taking too long to respond
    @pytest.mark.asyncio
    @pytest.mark.parametrize("server_cls, server_cb, to, expectation", TEST_BASE)
    async def test_base(self, server_cls, server_cb, to, expectation):
        token = uuid4()

        def callback(other_token, _):
            nonlocal token
            assert token == other_token

        server: t.Union[ShivaServer, ShivaServerAsync]

        server = server_cls(
            on_new_message_callback=server_cb,
            on_new_connection=partial(callback, token),
            on_connection_lost=partial(callback, token),
        )

        # if the server is sync, res will be None
        wfc_future = server.wait_for_connections(forever=False)

        # if the server is async, we need to await the result
        await wfc_future if wfc_future is not None else None

        # we close it and reopen it
        c_future = server.close()
        await c_future if c_future is not None else None
        wfc_future = server.wait_for_connections(forever=False)
        await wfc_future if wfc_future is not None else None

        # we create multiple clients to test the server with multiple connections
        cs = [await ShivaClientAsync.create_and_connect() for _ in range(13)]

        with expectation:
            c = sc.choice(cs)
            good_response = await c.send_message(self.GOOD_MESSAGE, timeout=to)
            assert good_response == self.GOOD_MESSAGE
            c = sc.choice(cs)
            empty_response = await c.send_message(self.EMPTY_MESSAGE, timeout=to)
            assert empty_response == self.EMPTY_MESSAGE

        [await client.disconnect() for client in cs]

        c_future = server.close()
        await c_future if c_future is not None else None

    # Here we test the server if it's blocking the loop
    @pytest.mark.asyncio
    @pytest.mark.parametrize("server_cls, server_cb, close, expectation", TEST_FOREVER)
    async def test_forever(self, server_cls, server_cb, close, expectation):
        server: t.Union[ShivaServer, ShivaServerAsync]

        server = server_cls(
            on_new_message_callback=server_cb,
        )

        async def server_thread():
            wfc_future = server.wait_for_connections(forever=True)
            try:
                return await wfc_future if wfc_future is not None else None
            except asyncio.CancelledError:
                return

        thread = threading.Thread(
            target=lambda: asyncio.run(server_thread()),
            daemon=True,
        )
        thread.start()

        # Wait for the server to accept connections
        trials = 0
        while trials < 100:
            try:
                client = await ShivaClientAsync.create_and_connect()
                break
            except Exception as _:
                trials += 1
            time.sleep(0.1)

        response = await client.send_message(self.GOOD_MESSAGE)
        assert response == self.GOOD_MESSAGE

        response = await client.send_message(self.EMPTY_MESSAGE)
        assert response == self.EMPTY_MESSAGE

        if close:
            c_future = server.close()
            await c_future if c_future is not None else None
            await client.disconnect()

        # check that the server is closed
        with expectation:
            client_2 = await ShivaClientAsync.create_and_connect()
            await client_2.disconnect()

        if not close:
            c_future = server.close()
            await c_future if c_future is not None else None
            await client.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "server_cls, server_cb, expectation, error", TEST_ERROR_LOGGING
    )
    async def test_error_logging(self, server_cls, server_cb, expectation, error):
        server: t.Union[ShivaServer, ShivaServerAsync]

        server = server_cls(
            on_new_message_callback=server_cb,
        )
        wfc_future = server.wait_for_connections(forever=False)
        await wfc_future if wfc_future is not None else None

        message = ShivaMessage()
        client = await ShivaClientAsync.create_and_connect()

        with expectation:
            response = await client.send_message(message)
            assert response == error

        c_future = server.close()
        await c_future if c_future is not None else None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("server_cls, server_cb, expectation", TEST_CLOSING)
    async def test_closing(self, server_cls, server_cb, expectation):
        server: t.Union[ShivaServer, ShivaServerAsync]
        server = server_cls(on_new_message_callback=server_cb)
        with expectation:
            c_future = server.close()
            await c_future if c_future is not None else None

    @pytest.mark.asyncio
    async def test_client_not_closed_sync(self):
        server = ShivaServer(on_new_message_callback=cb_sync)

        server.wait_for_connections(forever=False)

        client = await ShivaClientAsync.create_and_connect()
        good_response = await client.send_message(self.GOOD_MESSAGE)
        assert good_response == self.GOOD_MESSAGE
        if isinstance(server, ShivaServer):
            assert len(server._connections) == 1
            assert server._connections[0].fileno() != -1
            server._connections[0].shutdown(socket.SHUT_RDWR)
            server._connections[0].close()
            assert server._accepting_socket is not None
            assert server._accepting_socket.fileno() != -1
            server._accepting_socket.shutdown(socket.SHUT_RDWR)
            server._accepting_socket.close()
            server.close()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_client_not_closed_async(self):
        server = ShivaServerAsync(on_new_message_callback=cb_async)

        await server.wait_for_connections(forever=False)

        client = await ShivaClientAsync.create_and_connect()
        good_response = await client.send_message(self.GOOD_MESSAGE)
        assert good_response == self.GOOD_MESSAGE

        if isinstance(server, ShivaServerAsync):
            assert ShivaServerAsync._main_server is not None
            assert ShivaServerAsync._main_server.sockets is not None
            assert ShivaServerAsync._main_server.sockets[0].fileno() != -1
            await server.close()
            assert ShivaServerAsync._main_server is None

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_client_exceptions(self):
        client = ShivaClientAsync("localhost", 1234)
        with pytest.raises(ConnectionError):
            await client.disconnect()

        with pytest.raises(ConnectionError):
            await client.send_message(ShivaMessage())


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

        server = ShivaServer(on_new_message_callback=manage_message)

        server.wait_for_connections(forever=False)

        client = await ShivaClientAsync.create_and_connect()

        person = TestShivaBridge.Person(
            name="Alice",
            age=32,
            height=1.58,
            hair="brown",
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
