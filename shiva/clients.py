from __future__ import annotations

import asyncio

from loguru import logger

import shiva.model as sm
from shiva.messages import ShivaMessage


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
        if self._writer is None:
            err = "Can't disconnect, connect the client first."
            return logger.warning(err)

        self._writer.close()
        await self._writer.wait_closed()

    @classmethod
    async def create_and_connect(
        cls,
        host: str = "localhost",
        port: int = sm.ShivaConstants.DEFAULT_PORT,
    ) -> ShivaClientAsync:
        client = ShivaClientAsync(host, port)
        await client.connect()
        return client

    async def send_message(
        self, message: ShivaMessage, timeout: float = 0
    ) -> ShivaMessage:
        if self._writer is None or self._reader is None:
            err = "Can't send message, connect the client first."
            raise ValueError(err)

        await ShivaMessage.send_message_async(self._writer, message)
        logger.trace(f"Message sent: {message}")
        response_message = await ShivaMessage.receive_message_async(
            self._reader, timeout=timeout
        )
        logger.trace(f"Message received: {response_message}")
        response_message.sender = self._writer.get_extra_info("peername")
        return response_message
