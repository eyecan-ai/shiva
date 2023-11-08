# Shiva

<img src='docs/images/logo.jpeg' height=400 width=400 />


### Table of contents

* [Introduction](#introduction)
    * [Message](#message)
    * [Chunks](#chunks)
* [Data](#data)
    * [Metadata](#metadata)
    * [Namespace](#namespace-aka-command)
    * [Tensor](#tensor)



## Introduction

### Message

The **Shiva** Message is a binary flow of the following chunks:

<img src='docs/images/Message.png'/>

### Chunks

Message chunks are made as follows:

> :warning: All integers (**uint32**) in headers are encoded in **big endian** during send and receive.

<img src='docs/images/MessageZoom.png' />

<br>

> :warning: All integers (**uint32**) in headers are encoded in **big endian** during send and receive.

## Data

### Metadata

The metadata is a generic JSON string.

### Namespace (aka. Command)

The namespace is a generic string that can be used, for example, to route the message to the right handler.

### Tensor

Each tensor can be a multi-dimensional array of any type. Hence, the tensor has a **rank**, a **shape** and a **type**. The rank is the number of dimensions, the shape is a list of integers, one per dimension, and the type is a string that can be one of the following map:

```
{
    float16: 0,
    float32: 1,
    float64: 2,
    uint8: 3,
    int8: 4,
    uint16: 5,
    int16: 6,
    uint32: 7,
    int32: 8,
    uint64: 9,
    int64: 10,
    double: 11,
    longdouble: 12,
    longlong: 13,
    complex64: 14,
    complex128: 15,
    bool: 17,
}
```

For example a (h,w,c) image of 8-bit unsigned integers will have a rank of 3, a shape of [h,w,c] and a type of 3 (uint8).


