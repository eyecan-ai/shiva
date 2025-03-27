from __future__ import annotations

import asyncio
import json
import socket
import struct

import deepdiff
import numpy as np
import pydantic.v1 as pyd
from loguru import logger

from shiva.model import (
    CustomModel,
    DataPackaging,
    GlobalHeader,
    ShivaConstants,
    TensorHeader,
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
    tensors: list[np.ndarray] = pyd.Field(default_factory=list)

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

    def tensors_headers(self) -> list[TensorHeader]:
        """Builds the list of tensor headers"""

        headers = []

        for t in self.tensors:
            dt = t.dtype.newbyteorder(">").str  # force big endian
            headers.append(TensorHeader(tensor_rank=t.ndim, tensor_dtype=dt))

        return headers

    def tensors_shapes(self) -> list[list[int]]:
        """Returns the list of tensor shapes"""

        return [list(t.shape) for t in self.tensors]

    def tensors_data(self) -> list[bytes]:
        """Returns the list of tensors data as list of bytes"""
        data = []

        for t in self.tensors:
            be = t.astype(t.dtype.newbyteorder(">"))  # force big endian
            data.append(be.tobytes())

        return data

    def metadata_data(self) -> bytes:
        """Returns the metadata as bytes"""
        if len(self.metadata) == 0:
            return b""
        return self.json_metadata().encode("utf-8")

    def namespace_data(self) -> bytes:
        return self.namespace.encode("utf-8")

    def flush(self) -> bytes:
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
        tensors_headers: list[TensorHeader] = []
        tensor_shapes: list[list[int]] = []
        tensors: list[np.ndarray] = []

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
        try:
            data = cls._readexactly(connection, GlobalHeader.pack_size())
            global_header = GlobalHeader.unpack(data)
        except struct.error as e:
            err = "No data received, connection aborted"
            raise ConnectionAbortedError(err) from e

        # retrieve the following sizes
        metadata_size = global_header.metadata_size
        n_tensors = global_header.n_tensors
        tail_string_size = global_header.tail_string_size

        # receive the tensors headers
        tensors_headers: list[TensorHeader] = []
        tensor_shapes: list[list[int]] = []
        tensors: list[np.ndarray] = []

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
            expected_data = np.prod(shape, dtype=int) * bytes_per_element

            # receive the data
            data = cls._readexactly(connection, expected_data)
            # convert the data into a numpy array
            t = np.frombuffer(
                data,
                dtype=tensor_header.tensor_dtype,
            ).reshape(shape)

            # shiva sends tensors in big endian, convert them to the host endian
            t = t.astype(t.dtype.newbyteorder("="))

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
            message.tensors_data(),
        ):
            connection.send(h.pack())
            connection.send(DataPackaging.pack_ints(s))
            connection.send(t)

        # write the metadata and drain the buffer
        connection.send(message.metadata_data())

        # write the namespace and drain the buffer
        connection.send(message.namespace_data())

    @classmethod
    async def _readexactly_async(
        cls, reader: asyncio.StreamReader, payload_size: int, timeout: float
    ) -> bytes:
        """Reads exactly payload_size bytes from the reader

        Args:
            reader (StreamReader): the stream reader
            payload_size (int): the size of the payload
            timeout (float): the timeout in seconds to wait for the
            data (only if timeout > 0), if the timeout expires, an
            asyncio.TimeoutError exception is raised

        Returns:
            bytes: the payload read from the socket

        Raises:
            asyncio.TimeoutError: if the timeout expires
        """

        if timeout > 0:
            return await asyncio.wait_for(
                reader.readexactly(payload_size), timeout=timeout
            )
        else:
            return await reader.readexactly(payload_size)

    @classmethod
    async def receive_message_async(
        cls, reader: asyncio.StreamReader, timeout: float = 0
    ) -> ShivaMessage:
        """Receives a Shiva message from the reader

        Args:
            reader (StreamReader): the stream reader
            timeout (float): the timeout in seconds to wait for the
            data (only if timeout > 0), if the timeout expires, an
            asyncio.TimeoutError exception is raised

        Returns:
            ShivaMessage: the built Shiva message

        Raises:
            asyncio.TimeoutError: if the timeout expires
        """

        # receive the global header
        try:
            data = await cls._readexactly_async(
                reader, GlobalHeader.pack_size(), timeout
            )
            global_header = GlobalHeader.unpack(data)
        except asyncio.IncompleteReadError as e:
            err = "No data received, connection aborted"
            raise ConnectionAbortedError(err) from e

        logger.debug(f"Global header: {global_header}")

        # retrieve the following sizes
        metadata_size = global_header.metadata_size
        n_tensors = global_header.n_tensors
        tail_string_size = global_header.tail_string_size

        # receive the tensors headers
        tensors_headers: list[TensorHeader] = []
        tensor_shapes: list[list[int]] = []
        tensors: list[np.ndarray] = []

        for idx in range(n_tensors):
            # receive a single tensor header
            data = await cls._readexactly_async(
                reader, TensorHeader.pack_size(), timeout
            )
            tensor_header = TensorHeader.unpack(data)
            tensors_headers.append(tensor_header)
            logger.debug(f"Tensor [{idx}] header: {tensor_header}")

            # # receive the tensors shapes
            # the size of the shape is 4 bytes per dimension
            shape_size = 4 * tensor_header.tensor_rank

            # receive the shape
            data = await cls._readexactly_async(reader, shape_size, timeout)
            shape = DataPackaging.unpack_ints(data)
            tensor_shapes.append(shape)
            logger.debug(f"Tensor [{idx}] shape: {shape}")

            # receive the tensors data
            # the size of the data is the product of the shape elements times the size
            # of the tensor data type (byte-size)
            bytes_per_element = np.dtype(tensor_header.tensor_dtype).itemsize
            expected_data = np.prod(shape, dtype=int) * bytes_per_element
            logger.debug(f"Tensor [{idx}] expected data: {expected_data}")

            # receive the data
            data = await cls._readexactly_async(reader, expected_data, timeout)

            # convert the data into a numpy array
            t = np.frombuffer(
                data,
                dtype=tensor_header.tensor_dtype,
            ).reshape(shape)

            # shiva sends tensors in big endian, convert them to the host endian
            t = t.astype(t.dtype.newbyteorder("="))

            tensors.append(t)

        # receive the metadata if any
        logger.debug(f"Metadata expecting size: {metadata_size}")
        metadata = {}
        if metadata_size > 0:
            data = await cls._readexactly_async(reader, metadata_size, timeout)
            metadata = json.loads(data.decode("utf-8"))

        # receive the namespace if any
        logger.debug(f"Namespace expecting size: {tail_string_size}")
        namespace = ""
        if tail_string_size > 0:
            data = await cls._readexactly_async(reader, tail_string_size, timeout)
            namespace = data.decode("utf-8")

        logger.debug("Message complete, sending response")

        # return the built message
        return ShivaMessage(metadata=metadata, tensors=tensors, namespace=namespace)

    @classmethod
    async def send_message_async(
        cls,
        writer: asyncio.StreamWriter,
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
        logger.debug(f"Sent -> Global header: {message.global_header()}")
        await writer.drain()

        for h, s, t in zip(
            message.tensors_headers(),
            message.tensors_shapes(),
            message.tensors_data(),
        ):
            writer.write(h.pack())
            await writer.drain()
            writer.write(DataPackaging.pack_ints(s))
            await writer.drain()
            writer.write(t)
            await writer.drain()

        # write the metadata and drain the buffer
        writer.write(message.metadata_data())
        logger.debug(f"Sent -> Metadata: {message.metadata_data()}")
        await writer.drain()

        # write the namespace and drain the buffer
        writer.write(message.namespace_data())
        logger.debug(f"Sent -> Namespace: {message.namespace_data()}")
        await writer.drain()


class ShivaReservedMessage(ShivaMessage):
    """
    A private Shiva message that is used for internal communication within the
    Shiva framework. It is used to send special messages like errors, warnings,
    and other internal messages.
    """

    TAG = "reserved"

    class MetadataSchema(pyd.BaseModel):
        type: str
        message: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.namespace = f"{ShivaConstants.RESERVED_PREFIX}{self.TAG}"


class ShivaErrorMessage(ShivaReservedMessage):
    """
    A Shiva message that is used to send errors
    """

    TAG = "error"

    def __init__(self, exception: Exception, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.namespace = f"{ShivaConstants.RESERVED_PREFIX}{self.TAG}"
        self.metadata = self.MetadataSchema(
            type=exception.__class__.__name__,
            message=str(exception),
        ).dict()
