import asyncio
import socket
import threading
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
        self._alive = False
        self._accepting_socket: t.Optional[socket.socket] = None
        self._host = None
        self._port = None

        self._threads: list[threading.Thread] = []
        self._connections: list[socket.socket] = []  # To keep track of connections

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

        def accept_connections(socket: socket.socket):
            while self._alive:
                try:
                    connection, address = socket.accept()
                except OSError:
                    # When the socket is shutdown, the accept method raises an OSError,
                    # so we catch it and let the loop exit gracefully
                    logger.trace("Accepting socket closed.")
                else:
                    if self._on_new_connection is not None:
                        self._on_new_connection(address)

                    self._on_connection_callback(connection, address)

                    self._connections.append(connection)

        if forever:
            accept_connections(self._accepting_socket)
        else:
            self._accepting_thread = threading.Thread(
                target=accept_connections,
                args=(self._accepting_socket,),
                daemon=True,
            )
            self._accepting_thread.start()

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
                except OSError:  # Generic exception to catch IO errors from the socket
                    if self._on_connection_lost is not None:
                        self._on_connection_lost(address)
                except Exception as e:
                    logger.error(
                        f"{e.__class__.__name__}: {e.args[0] if e.args else ''}"
                    )
                    ShivaMessage.send_message(connection, ShivaErrorMessage(e))

        thread = threading.Thread(target=reading_loop, daemon=True)
        thread.start()

        self._threads.append(thread)

    def close(self):
        self._alive = False
        if self._accepting_socket is None:
            err = "Server is not running, did you forget to call wait_for_connections?"
            raise RuntimeError(err)
        if self._accepting_socket.fileno() != -1:
            self._accepting_socket.shutdown(socket.SHUT_RDWR)
            self._accepting_socket.close()
        for connection in self._connections:
            if connection.fileno() != -1:
                connection.shutdown(socket.SHUT_RDWR)
                connection.close()
        for thread in self._threads:
            thread.join()

        self._accepting_socket = None
        self._connections = []
        self._threads = []


class ShivaServerAsync:
    _main_server = None

    def __init__(
        self,
        on_new_message_callback: t.Callable[[ShivaMessage], t.Awaitable[ShivaMessage]],
        on_new_connection: t.Optional[t.Callable[[tuple], None]] = None,
        on_connection_lost: t.Optional[t.Callable[[tuple], None]] = None,
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
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # peername
        peername = writer.get_extra_info("peername")

        if self._on_new_connection is not None:
            self._on_new_connection(peername)

        while True:
            try:
                message = await ShivaMessage.receive_message_async(reader)

                response_message: ShivaMessage = await self._on_new_message_callback(
                    message
                )

                await ShivaMessage.send_message_async(writer, response_message)
            except (asyncio.exceptions.IncompleteReadError, BrokenPipeError):
                if self._on_connection_lost is not None:
                    self._on_connection_lost(peername)
                break
            except Exception as e:
                logger.error(f"{e.__class__.__name__}: {e.args[0] if e.args else ''}")
                await ShivaMessage.send_message_async(writer, ShivaErrorMessage(e))

    @classmethod
    async def accept_new_connections(
        cls,
        on_connection_callback: t.Callable,
        host: str = "0.0.0.0",
        port: int = ShivaConstants.DEFAULT_PORT,
        forever: bool = True,
    ) -> None:
        async def new_connection(reader, writer):
            await on_connection_callback(reader, writer)

        cls._main_server = await asyncio.start_server(
            new_connection,
            host,
            port,
            start_serving=False,
        )

        if forever:
            async with cls._main_server:
                await cls._main_server.serve_forever()
        else:
            await cls._main_server.start_serving()

    @classmethod
    async def close(cls):
        if cls._main_server is None:
            err = "Server is not running, did you forget to call wait_for_connections?"
            raise RuntimeError(err)
        cls._main_server.close()
        await cls._main_server.wait_closed()
        cls._main_server = None
