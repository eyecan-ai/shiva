from __future__ import annotations
import asyncio
import json
import socket
import struct
from abc import ABC, abstractmethod, abstractclassmethod
import threading
from typing import Callable, List, Optional
import numpy as np
import pydantic as pyd
from asyncio import StreamReader, StreamWriter
import deepdiff
from loguru import logger
import time


class ShivaConstants:
    DEFAULT_PORT = 6174
    TENSORS_KEY = '__tensors__'


class PackableHeader(ABC):
    """
    An abstract class for packable headers. It is used to define the
    generic interface for a packable headers
    """

    @abstractmethod
    def pack(self):
        """
        Packs the header into a bytes object
        """
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def unpack(cls, data) -> PackableHeader:
        """
        Unpacks the header from a bytes object and returns a new instance of the
        header
        """
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def pack_format(cls) -> str:
        """
        Returns the format string for the struct module. For example,
        if the header is composed of 3 integers and 2 unsigned bytes, the format string
        should be "iiiBB"

        Returns:
            str: the format string
        """
        raise NotImplementedError

    @classmethod
    def pack_size(cls) -> int:
        """
        Returns the size of the header in bytes, it can be computed from the
        pack_format string

        Returns:
            int: the size of the header in bytes
        """
        return struct.Struct(cls.pack_format()).size

    @classmethod
    @abstractclassmethod
    def magic_number(cls) -> Optional[tuple]:
        """
        Some headers may have a magic number, which is a tuple of bytes that is
        used to identify the header. The first header should be, like for files format,
        a custom magic number in order to perform signal coupling and avoid to read
        garbage data. If the header does not have a magic number, it should return None

        Returns:
            Optional[tuple]: the magic number, e.g. (6, 66, 11, 1)
        """
        raise NotImplementedError


class CustomModel(pyd.BaseModel):
    """A custom pydantic model that allows to use arbitrary types"""

    class Config:
        arbitrary_types_allowed = True


class DataPackaging:
    """
    Helper class for data packaging
    """

    @classmethod
    def pack_ints(cls, ints: List[int]) -> bytes:
        """Packs a list of integers into a bytes object

        Args:
            ints (List[int]):  the list of integers

        Returns:
            bytes: the corresponding bytes object
        """
        return struct.pack(f"!{len(ints)}i", *ints)

    @classmethod
    def unpack_ints(cls, data: bytes) -> List[int]:
        """Unpacks a bytes object into a list of integers

        Args:
            data (bytes): the bytes object

        Returns:
            List[int]:  the corresponding list of integers
        """

        if len(data) % 4 != 0:
            raise ValueError("The length of the data is not divisible by 4")

        return list(struct.unpack(f"!{len(data)//4}i", data))


class GlobalHeader(CustomModel, PackableHeader):
    """
    The global header of the message, it contains information to retrieve the
    whole message payload
    """

    # the size of the metadata in bytes
    metadata_size: int = pyd.Field(ge=0, le=2**32)

    # the number of tensors in the message
    n_tensors: int = pyd.Field(ge=0, le=256)

    # the size of the detail string in bytes
    tail_string_size: int = pyd.Field(default=0, ge=0, le=256)

    @classmethod
    def pack_format(cls) -> str:
        return "!BBBBiBBBB"

    @classmethod
    def magic_number(cls) -> Optional[tuple]:
        return (6, 66, 11, 1)

    @classmethod
    def _compute_crc(cls, elements: list) -> int:
        return sum(elements) % 256

    def pack(self):
        elements = [
            *(self.magic_number()),
            self.metadata_size,
            self.n_tensors,
            self.tail_string_size,
        ]
        elements.append(self._compute_crc(elements))
        elements.append(self._compute_crc(elements))
        return struct.pack(self.pack_format(), *elements)

    @classmethod
    def unpack(cls, data):
        elements = struct.unpack(cls.pack_format(), data)

        # check the crc(s)
        if elements[-2] != cls._compute_crc(elements[:-2]):
            raise ValueError("Wrong CRC 1")

        if elements[-1] != cls._compute_crc(elements[:-1]):
            raise ValueError("Wrong CRC 2")

        # check if the magic number is correct (if any
        mn = cls.magic_number()
        if mn is not None:
            # the first len(mn) elements should be the magic number and should be equal
            # to the magic number of the class
            if elements[: len(mn)] != cls.magic_number():
                raise ValueError("Wrong magic numbers")

        return cls(
            metadata_size=elements[4],
            n_tensors=elements[5],
            tail_string_size=elements[6],
        )


class TensorDataTypes:
    """
    Manifest of the supported tensor data types
    """

    # the dictionary that maps numpy types to the corresponding integer
    RAW_NUMPY_2_DTYPE = {
        np.float16: 0,
        np.float32: 1,
        np.float64: 2,
        np.uint8: 3,
        np.int8: 4,
        np.uint16: 5,
        np.int16: 6,
        np.uint32: 7,
        np.int32: 8,
        np.uint64: 9,
        np.int64: 10,
        np.double: 11,
        np.longdouble: 12,
        np.longlong: 13,
        np.complex64: 14,
        np.complex128: 15,
        np.bool_: 17,
    }

    # convert the keys of RAW_ NUMPY_2_DTYPE into numpy strings,
    # e.g. "<f4" instead of "float32"
    NUMPY_2_DTYPE = {np.dtype(k).str: v for k, v in RAW_NUMPY_2_DTYPE.items()}

    # create the inverse dictionary
    DTYPE_2_NUMPY = {v: k for k, v in NUMPY_2_DTYPE.items()}


class TensorHeader(CustomModel, PackableHeader):
    """
    The header of a tensor, it contains information to retrieve the tensor payload by
    packing the tensor rank and the tensor data type
    """

    # tensor rank, e.g. a tensor with shape (4,128,128,3) has rank 4
    tensor_rank: int = pyd.Field(ge=0, le=255)

    # tensor data type, e.g. float32 or uint8 or int64 etc. It is represented as the
    # numpy string representation of the data type, e.g. "<f4" is the numpy string
    # representation of float32, "<u1" is the numpy string representation of uint8 etc.
    tensor_dtype: str = pyd.Field(...)

    @classmethod
    def pack_format(cls) -> str:
        return "!BB"

    @classmethod
    def magic_number(cls) -> Optional[tuple]:
        return None

    def pack(self):
        return struct.pack(
            self.pack_format(),
            self.tensor_rank,
            TensorDataTypes.NUMPY_2_DTYPE[self.tensor_dtype],
        )

    @classmethod
    def unpack(cls, data):
        elements = struct.unpack(cls.pack_format(), data)
        tensor_rank, tensor_dtype = elements
        return cls(
            tensor_rank=tensor_rank,
            tensor_dtype=TensorDataTypes.DTYPE_2_NUMPY[tensor_dtype],
        )


class ShivaMessage(CustomModel):
    """
    The whole Shiva message, it is a simple object for the user which contains
    only the metadata and the tensors. It is used to serialize and deserialize the
    message.
    """

    # the generic user metadata
    metadata: dict = pyd.Field(default_factory=dict)

    # a list of tensors
    tensors: List[np.ndarray] = pyd.Field(default_factory=list)

    # namespace
    namespace: str = pyd.Field(default_factory=str)

    # sender
    sender: tuple = pyd.Field(default_factory=tuple)

    def __eq__(self, other):
        if not isinstance(other, ShivaMessage):
            return False

        if self.namespace != other.namespace:
            return False

        # deepdiff metadata
        if deepdiff.DeepDiff(self.metadata, other.metadata, ignore_order=True):
            return False

        if len(self.tensors) != len(other.tensors):
            return False

        for t1, t2 in zip(self.tensors, other.tensors):
            if not np.allclose(t1, t2):
                return False

        return True

    def json_metadata(self) -> str:
        """Returns the metadata as a json string"""

        return json.dumps(self.metadata)

    def global_header(self) -> GlobalHeader:
        """Builds the global header"""

        return GlobalHeader(
            metadata_size=len(self.metadata_data()),
            n_tensors=len(self.tensors),
            tail_string_size=len(self.namespace_data()),
        )

    def tensors_headers(self) -> List[TensorHeader]:
        """Builds the list of tensor headers"""

        return [
            TensorHeader(tensor_rank=t.ndim, tensor_dtype=t.dtype.str)
            for t in self.tensors
        ]

    def tensors_shapes(self) -> List[List[int]]:
        """Returns the list of tensor shapes"""

        return [list(t.shape) for t in self.tensors]

    def tensors_data(self) -> List[bytes]:
        """Returns the list of tensors data as list of bytes"""
        return [t.tobytes() for t in self.tensors]

    def metadata_data(self) -> bytes:
        """Returns the metadata as bytes"""
        if len(self.metadata) == 0:
            return b""
        return self.json_metadata().encode("utf-8")

    def namespace_data(self) -> bytes:
        return self.namespace.encode("utf-8")

    def flush(self) -> List[any]:
        buffer = []
        buffer.append(self.global_header().pack())

        for tensor_header, tensor_shape, tensor_data in zip(
            self.tensors_headers(),
            self.tensors_shapes(),
            self.tensors_data(),
        ):
            buffer.append(tensor_header.pack())
            buffer.append(DataPackaging.pack_ints(tensor_shape))
            buffer.append(tensor_data)

        buffer.append(self.metadata_data())
        buffer.append(self.namespace_data())

        # buffer is a list of bytes, transform it into a single bytes object
        return b"".join(buffer)

    @classmethod
    def parse(cls, buffer: bytes) -> ShivaMessage:
        global_header_chunk = buffer[: GlobalHeader.pack_size()]
        buffer = buffer[GlobalHeader.pack_size() :]
        global_header = GlobalHeader.unpack(global_header_chunk)

        metadata_size = global_header.metadata_size
        n_tensors = global_header.n_tensors

        # receive the tensors headers
        tensors_headers: List[TensorHeader] = []
        tensor_shapes: List[List[int]] = []
        tensors: List[np.ndarray] = []

        for _ in range(n_tensors):
            # receive a single tensor header
            tensor_header_chunk = buffer[: TensorHeader.pack_size()]
            buffer = buffer[TensorHeader.pack_size() :]

            tensor_header = TensorHeader.unpack(tensor_header_chunk)
            tensors_headers.append(tensor_header)

            # receive the tensors shapes
            # the size of the shape is 4 bytes per dimension
            shape_size = 4 * tensor_header.tensor_rank

            # receive the shape
            shape_data = buffer[:shape_size]
            buffer = buffer[shape_size:]
            shape = DataPackaging.unpack_ints(shape_data)
            tensor_shapes.append(shape)

            # receive the tensors data
            # the size of the data is the product of the shape elements times the size
            # of the tensor data type (byte-size)
            bytes_per_element = np.dtype(tensor_header.tensor_dtype).itemsize
            expected_data = np.prod(shape) * bytes_per_element

            # receive the data
            data = buffer[:expected_data]
            buffer = buffer[expected_data:]

            # convert the data into a numpy array
            t = np.frombuffer(
                data,
                dtype=tensor_header.tensor_dtype,
            ).reshape(shape)

            tensors.append(t)

        # receive the metadata if any
        metadata = {}
        if metadata_size > 0:
            data = buffer[:metadata_size]
            buffer = buffer[metadata_size:]
            metadata = json.loads(data.decode("utf-8"))

        # receive the namespace if any
        namespace = ""
        if global_header.tail_string_size > 0:
            data = buffer[: global_header.tail_string_size]
            namespace = data.decode("utf-8")

        return cls(metadata=metadata, tensors=tensors, namespace=namespace)

    @classmethod
    def _readexactly(cls, connection: socket.socket, payload_size: int) -> bytes:
        """Reads exactly payload_size bytes from the socket

        Args:
            connection (socket.socket): the socket
            payload_size (int): the size of the payload

        Returns:
            bytes: the payload read from the socket
        """

        received_size = 0
        received_data = b""
        while len(received_data) < payload_size:
            chunk = connection.recv(payload_size - received_size)
            if not chunk:
                break
            received_data += chunk
            received_size += len(chunk)
        return received_data

    @classmethod
    def receive_message(cls, connection: socket.socket) -> ShivaMessage:
        """Receives a Shiva message from the socket

        Args:
            connection (socket): the socket

        Returns:
            ShivaMessage: the built Shiva message
        """
        # receive the global header
        # receive the global header
        data = cls._readexactly(connection, GlobalHeader.pack_size())
        global_header = GlobalHeader.unpack(data)

        # retrieve the following sizes
        metadata_size = global_header.metadata_size
        n_tensors = global_header.n_tensors
        tail_string_size = global_header.tail_string_size

        # receive the tensors headers
        tensors_headers: List[TensorHeader] = []
        tensor_shapes: List[List[int]] = []
        tensors: List[np.ndarray] = []

        for _ in range(n_tensors):
            # receive a single tensor header
            data = cls._readexactly(connection, TensorHeader.pack_size())
            tensor_header = TensorHeader.unpack(data)
            tensors_headers.append(tensor_header)

            # receive the tensors shapes
            # the size of the shape is 4 bytes per dimension
            shape_size = 4 * tensor_header.tensor_rank

            # receive the shape
            data = cls._readexactly(connection, shape_size)
            shape = DataPackaging.unpack_ints(data)
            tensor_shapes.append(shape)

            # receive the tensors data
            # the size of the data is the product of the shape elements times the size
            # of the tensor data type (byte-size)
            bytes_per_element = np.dtype(tensor_header.tensor_dtype).itemsize
            expected_data = np.prod(shape) * bytes_per_element

            # receive the data
            data = cls._readexactly(connection, expected_data)
            # convert the data into a numpy array
            t = np.frombuffer(
                data,
                dtype=tensor_header.tensor_dtype,
            ).reshape(shape)
            tensors.append(t)

        # receive the metadata if any
        metadata = {}
        if metadata_size > 0:
            data = cls._readexactly(connection, metadata_size)
            metadata = json.loads(data.decode("utf-8"))

        # receive the namespace if any
        namespace = ""
        if tail_string_size > 0:
            data = cls._readexactly(connection, tail_string_size)
            namespace = data.decode("utf-8")

        # return the built message
        return ShivaMessage(metadata=metadata, tensors=tensors, namespace=namespace)

    @classmethod
    def send_message(cls, connection: socket.socket, message: ShivaMessage) -> None:
        """Sends a Shiva message to the socket

        Args:
            connection (socket): the socket
            message (ShivaMessage): the message to send
        """

        connection.send(message.global_header().pack())

        for h, s, t in zip(
            message.tensors_headers(),
            message.tensors_shapes(),
            message.tensors,
        ):
            connection.send(h.pack())
            connection.send(DataPackaging.pack_ints(s))
            connection.send(t.tobytes())

        # write the metadata and drain the buffer
        connection.send(message.metadata_data())

        # write the namespace and drain the buffer
        connection.send(message.namespace_data())

    @classmethod
    async def receive_message_async(cls, reader: StreamReader) -> ShivaMessage:
        """Receives a Shiva message from the reader

        Args:
            reader (StreamReader): the stream reader

        Returns:
            ShivaMessage: the built Shiva message
        """

        # receive the global header
        data = await reader.readexactly(GlobalHeader.pack_size())
        global_header = GlobalHeader.unpack(data)
        logger.debug(f"Global header: {global_header}")

        # retrieve the following sizes
        metadata_size = global_header.metadata_size
        n_tensors = global_header.n_tensors
        tail_string_size = global_header.tail_string_size

        # receive the tensors headers
        tensors_headers: List[TensorHeader] = []
        tensor_shapes: List[List[int]] = []
        tensors: List[np.ndarray] = []

        for idx in range(n_tensors):
            # receive a single tensor header
            data = await reader.readexactly(TensorHeader.pack_size())
            tensor_header = TensorHeader.unpack(data)
            tensors_headers.append(tensor_header)
            logger.debug(f"Tensor [{idx}] header: {tensor_header}")

            # # receive the tensors shapes
            # the size of the shape is 4 bytes per dimension
            shape_size = 4 * tensor_header.tensor_rank

            # receive the shape
            data = await reader.readexactly(shape_size)
            shape = DataPackaging.unpack_ints(data)
            tensor_shapes.append(shape)
            logger.debug(f"Tensor [{idx}] shape: {shape}")

            # receive the tensors data
            # the size of the data is the product of the shape elements times the size
            # of the tensor data type (byte-size)
            bytes_per_element = np.dtype(tensor_header.tensor_dtype).itemsize
            expected_data = np.prod(shape) * bytes_per_element
            logger.debug(f"Tensor [{idx}] expected data: {expected_data}")

            # receive the data
            data = await reader.readexactly(expected_data)

            # convert the data into a numpy array
            t = np.frombuffer(
                data,
                dtype=tensor_header.tensor_dtype,
            ).reshape(shape)

            tensors.append(t)

        # receive the metadata if any
        logger.debug(f"Metadata expecting size: {metadata_size}")
        metadata = {}
        if metadata_size > 0:
            data = await reader.readexactly(metadata_size)
            metadata = json.loads(data.decode("utf-8"))

        # receive the namespace if any
        logger.debug(f"Namespace expecting size: {tail_string_size}")
        namespace = ""
        if tail_string_size > 0:
            data = await reader.readexactly(tail_string_size)
            namespace = data.decode("utf-8")

        # return the built message
        return ShivaMessage(metadata=metadata, tensors=tensors, namespace=namespace)

    @classmethod
    async def send_message_async(
        cls,
        writer: StreamWriter,
        message: ShivaMessage,
    ) -> None:
        """Sends a Shiva message to the writer

        Args:
            writer (StreamWriter): the stream writer
            message (ShivaMessage): the message to send
        """

        # write the global header and drain the buffer (the drain operation lets the
        # asyncio to send the data to the other side and let other coroutines to run
        # in parallel
        writer.write(message.global_header().pack())
        await writer.drain()

        for h, s, t in zip(
            message.tensors_headers(),
            message.tensors_shapes(),
            message.tensors,
        ):
            writer.write(h.pack())
            await writer.drain()
            writer.write(DataPackaging.pack_ints(s))
            await writer.drain()
            writer.write(t.tobytes())
            await writer.drain()

        # write the metadata and drain the buffer
        writer.write(message.metadata_data())
        await writer.drain()

        # write the namespace and drain the buffer
        writer.write(message.namespace_data())
        await writer.drain()


class ShivaServer:
    _main_server = None

    def __init__(
        self,
        on_new_message_callback: Callable[[ShivaMessage], ShivaMessage],
        on_new_connection: Optional[Callable[[tuple], None]] = None,
        on_connection_lost: Optional[Callable[[tuple], None]] = None,
    ) -> None:
        self._on_new_message_callback = on_new_message_callback
        self._on_new_connection = on_new_connection
        self._on_connection_lost = on_connection_lost
        self._alive = False
        self._accepting_socket: Optional[socket.socket] = None
        self._host = None
        self._port = None

    @classmethod
    def _create_accepting_socket(
        cls,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
    ) -> socket.socket:
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = (host, port)
        sock.bind(server_address)
        sock.listen(1)
        return sock

    def wait_for_connections(
        self,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
        forever: bool = True,
    ):
        self._host = host
        self._port = port
        self._accepting_socket = self._create_accepting_socket(host=host, port=port)
        self._alive = True
        self._accepting_thread = None

        def accept_connections():
            while self._alive:
                connection, address = self._accepting_socket.accept()
                print("Killaccept...")

                self._on_connection_callback(connection, address)

                if self._on_new_connection is not None:
                    self._on_new_connection(address)

        if forever:
            accept_connections()
        else:
            self._accepting_thread = threading.Thread(
                target=accept_connections,
                daemon=True,
            )
            self._accepting_thread.start()

    def _on_connection_callback(
        self,
        connection: socket.socket,
        address: socket._RetAddress,
    ) -> None:
        def reading_loop():
            while self._alive:
                try:
                    message = ShivaMessage.receive_message(connection)
                    message.sender = address
                    response = self._on_new_message_callback(message)
                    ShivaMessage.send_message(connection, response)
                except Exception as e:
                    logger.error(e)
                    if self._on_connection_lost is not None:
                        self._on_connection_lost(address)
                    break

        thread = threading.Thread(target=reading_loop, daemon=True)
        thread.start()

    def close(self, wait_time: float = 0.5):
        self._alive = False
        try:
            # self connect to wakeup accepting socket
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                (self._host, self._port)
            )
            time.sleep(wait_time)
        except Exception as e:
            logger.warning(e)


class ShivaServerAsync:
    _main_server = None

    def __init__(
        self,
        on_new_message_callback: Callable[[ShivaMessage], ShivaMessage],
        on_new_connection: Optional[Callable[[tuple], None]] = None,
        on_connection_lost: Optional[Callable[[tuple], None]] = None,
    ) -> None:
        self._on_new_message_callback = on_new_message_callback
        self._on_new_connection = on_new_connection
        self._on_connection_lost = on_connection_lost

    async def wait_for_connections(
        self,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
        forever: bool = True,
    ):
        await self.accept_new_connections(
            self._on_connection_callback,
            host=host,
            port=port,
            forever=forever,
        )

    async def _on_connection_callback(
        self,
        reader: StreamReader,
        writer: StreamWriter,
    ) -> None:
        # peername
        peername = writer.get_extra_info("peername")

        logger.info(f"New connection <- {peername}")
        if self._on_new_connection is not None:
            self._on_new_connection(peername)

        while True:
            try:
                message = await ShivaMessage.receive_message_async(reader)

                message.sender = peername
                response_message: ShivaMessage = await self._on_new_message_callback(
                    message
                )

                await ShivaMessage.send_message_async(writer, response_message)
            except (asyncio.exceptions.IncompleteReadError, BrokenPipeError):
                logger.error(f"Connection lost <-> {peername}")
                if self._on_connection_lost is not None:
                    self._on_connection_lost(peername)
                break

    @classmethod
    async def accept_new_connections(
        cls,
        on_connection_callback: Callable,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
        forever: bool = True,
    ) -> None:
        async def new_connection(reader, writer):
            await on_connection_callback(reader, writer)

        cls._main_server = await asyncio.start_server(new_connection, host, port)

        if forever:
            async with cls._main_server:
                await cls._main_server.serve_forever()
        else:
            await cls._main_server.start_serving()

    @classmethod
    async def close(cls):
        if cls._main_server is not None:
            cls._main_server.close()
            await cls._main_server.wait_closed()


class ShivaClientAsync:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._reader, self._writer = None, None

    async def connect(self):
        self._reader, self._writer = await asyncio.open_connection(
            self._host,
            self._port,
        )

    async def disconnect(self):
        self._writer.close()
        await self._writer.wait_closed()

    @classmethod
    async def create_and_connect(
        cls,
        host: str = "localhost",
        port: int = ShivaConstants.DEFAULT_PORT,
    ) -> ShivaClientAsync:
        client = ShivaClientAsync(host, port)
        await client.connect()
        return client

    async def send_message(self, message: ShivaMessage) -> ShivaMessage:
        await ShivaMessage.send_message_async(self._writer, message)
        responose_message = await ShivaMessage.receive_message_async(self._reader)
        responose_message.sender = self._writer.get_extra_info("peername")
        return responose_message
