from __future__ import annotations

import abc
import typing
from ipaddress import IPv4Address, IPv6Address


class ByteSerializableObject(abc.ABC):
    """Parse and serialize objects to and from raw bytes."""

    __slots__ = ()

    @classmethod
    @abc.abstractmethod
    def parse(cls, data: bytes) -> typing.Self:
        """Parse an object from raw bytes."""

    @abc.abstractmethod
    def __bytes__(self) -> bytes:
        """Serialize the object to raw bytes."""

    def __str__(self) -> str:
        return self.__bytes__().decode()


class NetworkAddress(typing.NamedTuple):
    """Parse and serialize an address."""

    host: str | IPv4Address | IPv6Address
    port: int | None = None

    def __str__(self):
        if self.port and isinstance(self.host, IPv6Address):
            return f"[{self.host}]:{self.port}"
        elif self.port:
            return f"{self.host}:{self.port}"
        return str(self.host)

    @classmethod
    def parse(cls, data: str) -> NetworkAddress:
        if data.startswith("["):
            host, port = data[1:].split("]:")
            return cls(host=host, port=int(port))
        else:
            host, port = data.split(":")
            return cls(host=host, port=int(port))
