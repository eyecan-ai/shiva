# Simple Client Server

In this example we will emulate a simple client-server application that sends and receives messages. The client sends *metadata* and *tensors* to the server, and the server returns a response with the same *metadata* and *tensors* but increasing by 1 the values of the *tensors* named **counter** and
the field of metadata named **counter**. 

This is a simple showcase of the capabilities of Shiva which is a low-level transport layer that can be used to build more complex applications. For example, the use of `shv.ShivaConstants.TENSORS_KEY` in
metadata field to assign a name to a tensor is not mandatory, but it is a good practice to use it to identify the tensors in the message.

To run the example, first start the server:

```bash
python server.py
```

Then, in another terminal, run the client:

```bash
python client.py
```

You should see in the client's terminal a continuous stream of messages like this:

```bash
|
Roundtrip time: 29.01 ms Hz: 34.48
{'__tensors__': ['left', 'right', 'counter'], 'counter': 52}
[52] int64
|
```

Where **52** is the value of the incrementing counter in the server.