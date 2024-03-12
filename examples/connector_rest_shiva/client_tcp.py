import asyncio

import rich


async def send_bytes_to_server(host, port, data):
    """
    Sends a sequence of bytes to a server specified by host and port.

    Args:
    - host: The server's hostname or IP address as a string.
    - port: The port number on which the server is listening.
    - data: The byte sequence to send, as a bytes object.
    """
    _, writer = await asyncio.open_connection(host, port)

    try:
        writer.write(data)
        await writer.drain()
        rich.print(f"Successfully sent data to {host}:{port}")
    except Exception as e:
        rich.print(f"An error occurred: {e}")
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    # Example usage
    host = "127.0.0.1"  # Server's IP address
    port = 6174  # Server's port number

    global_header = bytes.fromhex("06420b0100000022000d8306")
    metadata_data_pre = bytes.fromhex("7b226964223a2022")
    metadata_data_var = [
        "363563633761393132373533353961313961393531363932",  # product
        "363563633761613632373533353961313961393531366439",  # marker
    ]
    metadata_data_post = bytes.fromhex("227d")
    namespace_data = bytes.fromhex("6368616e67652d666f726d6174")

    i = 0
    data = b"".join(
        [
            global_header,
            metadata_data_pre,
            bytes.fromhex(metadata_data_var[1]),
            metadata_data_post,
            namespace_data,
        ]
    )

    asyncio.run(send_bytes_to_server(host, port, data))
