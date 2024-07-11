from __future__ import annotations

import struct
import typing as t
from abc import ABC, abstractmethod

import numpy as np
import pydantic as pyd


class CustomModel(pyd.BaseModel):
    """A custom pydantic model that allows to use arbitrary types"""

    class Config:
        arbitrary_types_allowed = True


class ShivaConstants:
    DEFAULT_PORT = 6174
    TENSORS_KEY = "__tensors__"
    RESERVED_PREFIX = "!"


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
    @abstractmethod
    def unpack(cls, data) -> PackableHeader:
        """
        Unpacks the header from a bytes object and returns a new instance of the
        header
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
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
    @abstractmethod
    def magic_number(cls) -> t.Optional[tuple]:
        """
        Some headers may have a magic number, which is a tuple of bytes that is
        used to identify the header. The first header should be, like for files format,
        a custom magic number in order to perform signal coupling and avoid to read
        garbage data. If the header does not have a magic number, it should return None

        Returns:
            Optional[tuple]: the magic number, e.g. (6, 66, 11, 1)
        """
        raise NotImplementedError


class DataPackaging:
    """
    Helper class for data packaging
    """

    @classmethod
    def pack_ints(cls, ints: list[int]) -> bytes:
        """Packs a list of integers into a bytes object

        Args:
            ints (list[int]):  the list of integers

        Returns:
            bytes: the corresponding bytes object
        """
        return struct.pack(f"!{len(ints)}i", *ints)

    @classmethod
    def unpack_ints(cls, data: bytes) -> list[int]:
        """Unpacks a bytes object into a list of integers

        Args:
            data (bytes): the bytes object

        Returns:
            list[int]:  the corresponding list of integers
        """

        if len(data) % 4 != 0:
            err = f"The length of the data is not divisible by 4: {len(data)}"
            raise ValueError(err)

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
    def magic_number(cls) -> t.Optional[tuple]:
        return (6, 66, 11, 1)

    @classmethod
    def _compute_crc(cls, elements: list) -> int:
        return sum(elements) % 256

    def pack(self):
        magic_number = self.magic_number()
        elements = [
            *(magic_number if magic_number is not None else []),
            self.metadata_size,
            self.n_tensors,
            self.tail_string_size,
        ]
        elements.append(self._compute_crc(elements))
        elements.append(self._compute_crc(elements))
        return struct.pack(self.pack_format(), *elements)

    @classmethod
    def unpack(cls, data):  # TODO: struct.error
        elements = t.cast(list, struct.unpack(cls.pack_format(), data))

        # check the crc(s)
        if elements[-2] != cls._compute_crc(elements[:-2]):
            err = f"Wrong CRC 1: {elements[-2]} != {cls._compute_crc(elements[:-2])}"
            raise ValueError(err)

        if elements[-1] != cls._compute_crc(elements[:-1]):
            err = f"Wrong CRC 2: {elements[-1]} != {cls._compute_crc(elements[:-1])}"
            raise ValueError(err)

        # check if the magic number is correct (if any
        mn = cls.magic_number()
        if mn is not None:
            # the first len(mn) elements should be the magic number and should be equal
            # to the magic number of the class
            if elements[: len(mn)] != cls.magic_number():
                err = f"Wrong magic numbers: {elements[: len(mn)]} != {cls.magic_number()}"
                raise ValueError(err)

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
    # (np.float64 has been removed because it was overwritten by np.double,
    # thus the id 2 was never present in the dictionary)
    RAW_NUMPY_2_DTYPE: t.ClassVar = {
        np.dtype(k).newbyteorder(">"): v
        for k, v in [
            (np.float16, 0),
            (np.float32, 1),
            # (np.float64, 2),
            (np.uint8, 3),
            (np.int8, 4),
            (np.uint16, 5),
            (np.int16, 6),
            (np.uint32, 7),
            (np.int32, 8),
            (np.uint64, 9),
            (np.int64, 10),
            (np.double, 11),
            (np.longdouble, 12),
            (np.longlong, 13),
            (np.complex64, 14),
            (np.complex128, 15),
            (np.bool_, 17),
        ]
    }

    # convert the keys of RAW_ NUMPY_2_DTYPE into numpy strings
    # (e.g. ">f4" instead of "float32")
    NUMPY_2_DTYPE: t.ClassVar = {k.str: v for k, v in RAW_NUMPY_2_DTYPE.items()}

    # create the inverse dictionary
    DTYPE_2_NUMPY: t.ClassVar = {v: k for k, v in NUMPY_2_DTYPE.items()}
    # add id=2 for backwards compatibility, since np.float64 has been removed
    DTYPE_2_NUMPY[2] = np.dtype(np.double).newbyteorder(">").str


class TensorHeader(CustomModel, PackableHeader):
    """
    The header of a tensor, it contains information to retrieve the tensor payload by
    packing the tensor rank and the tensor data type
    """

    # tensor rank, e.g. a tensor with shape (4,128,128,3) has rank 4
    tensor_rank: int = pyd.Field(ge=0, le=255)

    # tensor data type, e.g. float32 or uint8 or int64 etc. It is represented as the
    # numpy string representation of the data type, e.g. ">f4" is the numpy string
    # representation of float32, ">u1" is the numpy string representation of uint8 etc.
    tensor_dtype: str = pyd.Field(...)

    @classmethod
    def pack_format(cls) -> str:
        return "!BB"

    @classmethod
    def magic_number(cls) -> t.Optional[tuple]:
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
