# Rest-Shiva Connector Demo

In this example we will show a connector bridge from an already running rest api server to a shiva client. The client uses the **namespace** field of the message to specify the **command** to be executed. For example:

* **INFO**: will return the information of the server.
* **INFERENCE**: will return the result of the inference.
* **BAD**: will represent an unknown command and the server will return an error.

To run the example, first start your server connector:

```bash
python server.py
```

To change host and port you can set the env variables:
```bash
HOST=localhost PORT=9999 python server.py
```

Then, in another terminal, run the client:

```bash
python client.py
```

You should see this output:

```bash
INFO
{'name': 'Rest/Shiva Connector', 'version': '0.0.1', 'vision_pipelines': ['camera']}

INFERENCE
ShivaMessage(
    metadata={},
    tensors=[
       ...
    ],
    namespace='inference',
    sender=('127.0.0.1', 6174)
)

BAD
ShivaMessage(metadata={}, tensors=[], namespace='error', sender=('127.0.0.1', 6174))
```

In the inference response message's field **tensors** you can see the matrix that was generated by the server
with a format like:

<table>
    <tr>
        <th>class</th>
        <th>score</th>
        <th>bbox_x</th>
        <th>bbox_y</th>
        <th>bbox_w</th>
        <th>bbox_h</th>
        <th>bbox_angle</th>
        <th>pose_3d_11</th>
        <th>pose_3d_12</th>
        <th>...</th>
        <th>pose_3d_33</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0.99</td>
        <td>0.01</td>
        <td>0.02</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>15.1</td>
        <td>0.99</td>
        <td>0</td>
        <td>0</td>
        <td>...</td>
        <td>1</td>
    </tr>
</table>

## :whale2: Docker

To build the image:
``` console
cd shiva/
docker build -f dockerfiles/Dockerfile -t shiva:latest .
```

To launch the service:

``` console
docker compose -f dockerfiles/docker-compose.shiva-rest-connector.yml up -d
```

To override environment variables you can create a  dockerfiles/.env and add the variables
you want to override.

If you need to communicate with the host on windows you can set the 
`EYENODE_API_HOST=host.docker.internal` in the .env and you will also have 
to remove network_mode: host from the compose file to be able to use the ports info.

 

