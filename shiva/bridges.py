from __future__ import annotations

import typing as t
from abc import ABC

import numpy as np

from shiva.messages import ShivaMessage
from shiva.model import CustomModel

TShivaBridge = t.TypeVar("TShivaBridge", bound="ShivaBridge")


class ShivaBridge(ABC, CustomModel):
    """Bridge between Pydantic models and Shiva messages.

    This class is used to convert a Pydantic model into a Shiva message and vice versa.
    The schema of the model is saved into the metadata of the Shiva message. The values
    with primitive types and strings are also saved directly into the metadata, while
    tensors (i.e., numpy arrays) are saved into the tensors list of the Shiva message
    and a special placeholder is used to reference the tensors in the metadata.

    Example:
        >>> class MyModel(ShivaBridge):
        ...     name: str
        ...     age: int
        ...     scores: np.ndarray
        ...
        >>> m = MyModel(name="shiva", age=10, scores=np.random.rand(3, 4))
        >>> msg = m.to_shiva_message(namespace="my_shiva_msg")
        >>> print(msg)
        ShivaMessage(
            metadata={'name': 'shiva', 'age': 10, 'scores': '__tensor__0'},
            tensors=[
                array([[0.37199876, 0.61100295, 0.42818011, 0.48479924],
            [0.926789  , 0.20982739, 0.78553886, 0.50265671],
            [0.85282731, 0.66210649, 0.01439065, 0.57840516]])
            ],
            namespace='my_shiva_msg',
            sender=()
        )
        >>> m2 = MyModel.from_shiva_message(msg)
        >>> b1, b2 = pickle.dumps(m), pickle.dumps(m2)
        >>> b1 == b2
        True
    """

    TENSOR: t.ClassVar[str] = "__tensor__"
    RECURSION: t.ClassVar[str] = "__recursion__"

    def to_shiva_message(self, namespace: str = "") -> ShivaMessage:
        """Convert the model into a Shiva message"""

        def parse(d: t.Mapping, tensor_start: int = 0) -> tuple[dict, list]:
            metadata = {}
            tensors = []

            tidx = tensor_start

            for k, v in d.items():
                if v is None:
                    metadata[k] = "null"
                elif isinstance(v, (int, float, str, bool)):
                    metadata[k] = v
                elif isinstance(v, t.Mapping):
                    metadata[k], ts = parse(v, tidx)
                    tensors.extend(ts)
                    tidx += len(ts)
                elif isinstance(v, t.Sequence):
                    ms = []
                    for i in range(len(v)):
                        m, ts = parse({self.RECURSION: v[i]}, tidx)
                        ms.append(m[self.RECURSION])
                        tensors.extend(ts)
                        tidx += len(ts)
                    metadata[k] = ms
                elif isinstance(v, np.ndarray):
                    tensors.append(v)
                    metadata[k] = f"{self.TENSOR}{tidx}"
                    tidx += 1
                else:
                    msg = f"ShivaBridge unsupported type {type(v)}"
                    raise ValueError(msg)

            return metadata, tensors

        m, ts = parse(self.dict())

        return ShivaMessage(metadata=m, tensors=ts, namespace=namespace)

    @classmethod
    def from_shiva_message(cls: type[TShivaBridge], msg: ShivaMessage) -> TShivaBridge:
        """Convert a Shiva message into the model

        Args:
            msg: the Shiva message.

        Returns:
            A new instance of the current ShivaBridge subclass.
        """

        def build(d: dict, tensors: list[np.ndarray]) -> dict:
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = build(v, tensors)
                elif isinstance(v, list):
                    for i in range(len(v)):
                        b = build({cls.RECURSION: v[i]}, tensors)
                        v[i] = b[cls.RECURSION]
                elif isinstance(v, str):
                    if v == "null":
                        d[k] = None
                    if v.startswith(cls.TENSOR):
                        idx = int(v.split(cls.TENSOR)[1])
                        # force native byte order since big endian is not supported
                        # in some libraries (e.g. pytorch)
                        tensor = tensors[idx]
                        d[k] = tensor.astype(tensor.dtype.newbyteorder("="))
            return d

        obj = build(msg.metadata, msg.tensors)
        return cls.parse_obj(obj)
