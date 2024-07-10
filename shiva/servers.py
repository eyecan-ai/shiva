import asyncio
import socket
import threading
import time
import typing as t

from loguru import logger

from shiva.messages import ShivaErrorMessage, ShivaMessage
from shiva.model import ShivaConstants


class ShivaServer:

    def __init__(
        self,
        on_new_message_callback: t.Callable[[ShivaMessage], ShivaMessage],
        on_new_connection: t.Optional[t.Callable[[tuple], None]] = None,
        on_connection_lost: t.Optional[t.Callable[[tuple], None]] = None,
    ) -> None:
        self._on_new_message_callback = on_new_message_callback
        self._on_new_connection = on_new_connection
        self._on_connection_lost = on_connection_lost

        # Threading management
        self._alive = False
        self._threads: list[threading.Thread] = []

        # Socket management
        self._accepting_socket: t.Optional[socket.socket] = None
        self._lock = threading.RLock()
        with self._lock:
            self._connections: list[socket.socket] = []

    def wait_for_connections(
        self,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
        forever: bool = True,
    ):
        ################
        # Server Setup #
        ################

        self._accepting_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._accepting_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._accepting_socket.bind((host, port))
        self._accepting_socket.listen(1)

        ###############
        # Server Loop #
        ###############

        self._alive = True

        if forever:
            self._accept_connections(self._accepting_socket)
        else:
            threading.Thread(
                target=self._accept_connections,
                args=(self._accepting_socket,),
                daemon=True,
            ).start()

    def _on_connection_callback(
        self,
        connection: socket.socket,
        address: tuple[str, int],
    ) -> None:
        def reading_loop():
            while self._alive:
                try:
                    message = ShivaMessage.receive_message(connection)
                    message.sender = address
                    response = self._on_new_message_callback(message)
                    ShivaMessage.send_message(connection, response)
                # Generic exception to catch IO errors from the socket, this is triggered
                # also when the client disconnects, so we can handle it
                except OSError:
                    if self._on_connection_lost is not None:
                        self._on_connection_lost(address)

                        # Client is disconnected, we remove it from the list
                        with self._lock:
                            self._connections.remove(connection)

                        break
                except Exception as e:
                    logger.error(
                        f"{e.__class__.__name__}: {e.args[0] if e.args else ''}"
                    )
                    ShivaMessage.send_message(connection, ShivaErrorMessage(e))

        if self._on_new_connection is not None:
            self._on_new_connection(address)

        thread = threading.Thread(target=reading_loop, daemon=True)
        thread.start()

        self._threads.append(thread)

    def _accept_connections(self, reader: socket.socket):

        while self._alive:
            try:
                writer, address = reader.accept()
            except OSError:
                # When the socket is shutdown, the accept method raises an OSError,
                # so we catch it and let the loop exit gracefully
                logger.trace("Accepting socket closed.")
            else:
                self._on_connection_callback(writer, address)

                with self._lock:
                    self._connections.append(writer)

    def close(self, wait_time: float = 0.1):
        self._alive = False
        if self._accepting_socket is None:
            err = "Server is not running, did you forget to call wait_for_connections?"
            raise RuntimeError(err)

        time.sleep(wait_time)

        # We close the accepting socket and all the connections
        logger.trace("Shutting down, closing sockets...")
        for sock in [self._accepting_socket, *self._connections]:
            try:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            except OSError:
                logger.trace(f"Socket {sock} already closed, skipping...")

        for thread in self._threads:
            thread.join()

        self._accepting_socket = None
        with self._lock:
            self._connections = []
        self._threads = []


class ShivaServerAsync:

    def __init__(
        self,
        on_new_message_callback: t.Callable[[ShivaMessage], t.Awaitable[ShivaMessage]],
        on_new_connection: t.Optional[t.Callable[[tuple], None]] = None,
        on_connection_lost: t.Optional[t.Callable[[tuple], None]] = None,
    ) -> None:
        self._on_new_message_callback = on_new_message_callback
        self._on_new_connection = on_new_connection
        self._on_connection_lost = on_connection_lost

        # Tasks management
        self._alive = False

        # Server management
        self._main_server = None

    async def wait_for_connections(
        self,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
        forever: bool = True,
    ):
        ################
        # Server Setup #
        ################

        self._main_server = await asyncio.start_server(
            self._accept_connections,  # type: ignore
            host,
            port,
            start_serving=False,
        )

        ###############
        # Server Loop #
        ###############

        self._alive = True

        if forever:
            async with self._main_server:
                await self._main_server.serve_forever()
        else:
            await self._main_server.start_serving()

    async def _on_connection_callback(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        async def reading_loop():
            while self._alive:
                try:
                    message = await ShivaMessage.receive_message_async(reader)

                    response_message: ShivaMessage = (
                        await self._on_new_message_callback(message)
                    )

                    await ShivaMessage.send_message_async(writer, response_message)
                except OSError:
                    if self._on_connection_lost is not None:
                        self._on_connection_lost(writer.get_extra_info("peername"))
                    break
                except Exception as e:
                    logger.error(
                        f"{e.__class__.__name__}: {e.args[0] if e.args else ''}"
                    )
                    await ShivaMessage.send_message_async(writer, ShivaErrorMessage(e))

        if self._on_new_connection is not None:
            self._on_new_connection(writer.get_extra_info("peername"))

        await reading_loop()

    async def _accept_connections(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:

        await self._on_connection_callback(reader=reader, writer=writer)

    async def close(self, wait_time: float = 0.1):
        self._alive = False
        if self._main_server is None:
            err = "Server is not running, did you forget to call wait_for_connections?"
            raise RuntimeError(err)

        await asyncio.sleep(wait_time)

        # We close the server and all the connections
        self._main_server.close()
        await self._main_server.wait_closed()
        self._main_server = None
