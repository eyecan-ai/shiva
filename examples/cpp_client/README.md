# C++ Client

In this example we will emulate a simple client-server application where the client is 
implemented in C++ and the server in Python. The client sends *metadata* and *tensors* to the server,
and the server returns a response with the same *metadata* and *tensors* but increasing by 1 the values
of the *tensors* named **counter** and the field of metadata named **counter**. 

For more details see the [README](../simple_client_server/README.md) of the Python client-server example.

## Launch the server first!

Launch the python server first:

```bash
python server.py
```

## Build & Run C++

Low level details on how to build the C++ client is out of the scope of this tutorial. You
can run a simple **cmake** building pipeline like this:

```bash
mkdir build
cd build
cmake ..
make
```

Then:

```bash
./shiva_client 127.0.0.1 6174
```

You should see a stream of messages like this:

```bash
|
FPS: 61
metadata: {"__tensors__":["tensor_1","tensor_2","counter","tensor_4"],"counter":102}
|
```