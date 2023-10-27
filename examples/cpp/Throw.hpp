#include "nlohmann/json.hpp"
#include <algorithm>
#include <arpa/inet.h> /* for sockaddr_in and inet_addr() */
#include <cstdint>
#include <cstring>
#include <iostream>
#include <netinet/tcp.h>
#include <stdio.h>  /* for printf() and fprintf() */
#include <stdlib.h> /* for atoi() and exit() */
#include <string.h> /* for memset() */
#include <string>
#include <sys/socket.h> /* for socket(), connect(), send(), and recv() */
#include <sys/time.h>
#include <unistd.h> /* for close() */
#define COMMAND_SIZE 32
typedef uint8_t byte;

namespace throwprotocol {

/**
 * DIEs
 */
void die(std::string errorMessage) {
  std::cout << errorMessage << std::endl;
  exit(1);
}
struct MessageHeader {
  uint8_t MAGIC[4];
  uint32_t metadata_size;
  uint8_t n_tensors;
  uint8_t trail_size;
  uint8_t CRC;
  uint8_t CRC2;

  MessageHeader() {}
  MessageHeader(int metadata_size, uint8_t n_tensors, uint8_t trail_size) {
    // Control Code
    this->MAGIC[0] = 6;
    this->MAGIC[1] = 66;
    this->MAGIC[2] = 11;
    this->MAGIC[3] = 1;

    // Payload Size
    this->metadata_size = metadata_size;
    this->trail_size = trail_size;
    this->n_tensors = n_tensors;

    // crc is sum module 256 of all values
    this->CRC = 0;
    this->CRC += this->MAGIC[0];
    this->CRC += this->MAGIC[1];
    this->CRC += this->MAGIC[2];
    this->CRC += this->MAGIC[3];
    this->CRC += this->metadata_size;
    this->CRC += this->n_tensors;
    this->CRC += this->trail_size;
    this->CRC = this->CRC % 256;
    this->CRC2 = this->CRC;
  }

  void receiveFromSocket(int sock) {
    if (recv(sock, this, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Receive Data Fails!");
  }
};

struct TensorHeader {
  uint8_t rank = 3;
  uint8_t dtype = 3;

  void receiveFromSocket(int sock) {
    if (recv(sock, this, sizeof(TensorHeader), 0) != sizeof(TensorHeader))
      die("Receive Data Fails!");
  }
};

class TensorF {
public:
  std::vector<float> data;
  std::vector<int> shape;

  TensorHeader getHeader() {
    TensorHeader header;
    header.rank = this->shape.size();
    header.dtype = 1;
    return header;
  }

  void sendHeader(int sock) {
    TensorHeader header = this->getHeader();
    if (send(sock, &header, sizeof(TensorHeader), 0) != sizeof(TensorHeader))
      die("Send Tensor Header fails!");
  }

  void sendShape(int sock) {
    if (send(sock, &this->shape[0], sizeof(int) * this->shape.size(), 0) !=
        sizeof(int) * this->shape.size())
      die("Send Tensor Size fails!");
  }

  void sendData(int sock) {
    if (send(sock, &this->data[0], sizeof(float) * this->data.size(), 0) !=
        sizeof(float) * this->data.size())
      die("Send Tensor Data fails!");
  }

  void receiveShape(int sock, int rank) {
    std::vector<int> shape;
    for (int i = 0; i < rank; i++) {
      int shape_element;
      if (recv(sock, &shape_element, sizeof(int), 0) != sizeof(int))
        die("Receive Data Fails!");
      shape.push_back(shape_element);
    }
    this->shape = shape;
  }

  void receiveFromSocket(int sock) {

    int expected_size = 1;
    // exptected size is product of all shape elements * 4 bytes
    for (int i = 0; i < this->shape.size(); i++) {
      expected_size *= this->shape[i];
    }
    expected_size *= 4;

    std::shared_ptr<float> response_array(new float[expected_size],
                                          std::default_delete<float[]>());

    float *received_data = response_array.get();
    int received_size = 0;
    while (received_size < expected_size) {
      int remains = expected_size - received_size;
      int chunk_size = recv(sock, &(received_data)[received_size], remains, 0);
      received_size += chunk_size;
    }

    this->data = std::vector<float>(response_array.get(),
                                    response_array.get() + expected_size);

    // delete received_data array
    // delete[] received_data;
  }
};

class Message {
public:
  nlohmann::json metadata;
  std::string command;
  std::vector<TensorF> tensors;
  Message() : metadata(nlohmann::json::object()), command(""), tensors() {}

  MessageHeader receiveHeader(int sock) {
    MessageHeader header;
    if (recv(sock, &header, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Receive Data Fails!");
    return header;
  }

  void receiveMetadata(int sock, int metadata_size) {
    std::shared_ptr<uint8_t> response_array(new uint8_t[metadata_size],
                                            std::default_delete<uint8_t[]>());

    uint8_t *received_data = response_array.get();
    int received_size = 0;
    while (received_size < metadata_size) {
      int remains = metadata_size - received_size;
      int chunk_size = recv(sock, &(received_data)[received_size], remains, 0);
      received_size += chunk_size;
    }

    std::string response_string((char *)response_array.get(), metadata_size);
    this->metadata = nlohmann::json::parse(response_string);
  }

  void receiveCommand(int sock, int trail_size) {
    std::shared_ptr<uint8_t> response_array(new uint8_t[trail_size],
                                            std::default_delete<uint8_t[]>());

    uint8_t *received_data = response_array.get();
    int received_size = 0;
    while (received_size < trail_size) {
      int remains = trail_size - received_size;
      int chunk_size = recv(sock, &(received_data)[received_size], remains, 0);
      received_size += chunk_size;
    }

    std::string response_string((char *)response_array.get(), trail_size);
    this->command = response_string;
  }

  void sendHeader(int sock) {
    MessageHeader header(this->command.size(), this->tensors.size(),
                         this->metadata.dump().size());
    if (send(sock, &header, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Send Header fails!");
  }

  void sendMetadata(int sock) {
    std::cout << "Real seding" << this->metadata.dump() << std::endl;
    if (send(sock, this->metadata.dump().c_str(), this->metadata.dump().size(),
             0) != this->metadata.dump().size())
      die("Send Metadata fails!");
  }

  void sendCommand(int sock) {
    if (send(sock, this->command.c_str(), this->command.size(), 0) !=
        this->command.size())
      die("Send Command fails!");
  }
};

class ThrowClientExample {

protected:
  /* Socket parameters */
  int sock;                         /* socket handle */
  struct sockaddr_in echoServAddr;  /* server address */
  unsigned short serverPort = 8000; /* server port */
  std::string server_address;       /* Server IP address (dotted quad) */
  unsigned short port;              /* Input Image */

public:
  ThrowClientExample(std::string server_address, unsigned short port) {

    this->server_address = server_address;
    this->port = port;

    /* TCP Socket Creation */
    if ((sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
      die("socket() initialization failed");

    /* Server Address Structure */
    memset(&echoServAddr, 0, sizeof(echoServAddr)); /* Zero out structure */
    echoServAddr.sin_family = AF_INET; /* Internet address family */
    echoServAddr.sin_addr.s_addr =
        inet_addr(this->server_address.c_str()); /* Server IP address */
    echoServAddr.sin_port = htons(this->port);   /* Server port */

    /* Connection */
    if (connect(sock, (struct sockaddr *)&echoServAddr, sizeof(echoServAddr)) <
        0)
      die("connect() failed");

    /* TCP FAST NO DELAY mode */
    int enable_no_delay = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &enable_no_delay,
                   sizeof(int)) < 0)
      die("TCP_NODELAY failed");

    /* Handshake wait... */
    usleep(10000);
  }

  /**
   * Sends and HEADER through the socket
   */
  void sendHeader(int sock, MessageHeader &header) {
    if (send(sock, &header, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Send Header fails!");
  }

  /**
   * Receives an HEADER through the socket
   */
  void receiveHeader(MessageHeader &header) {
    if (recv(sock, &header, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Receive Data Fails!");
  }

  Message sendAndReceiveMessage(Message &message) {

    MessageHeader header(message.metadata.dump().size(), message.tensors.size(),
                         message.command.size());

    std::cout << "Sending" << message.metadata << std::endl;
    if (send(sock, &header, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Send Header fails!");

    for (int i = 0; i < message.tensors.size(); i++) {
      message.tensors[i].sendHeader(sock);
    }

    for (int i = 0; i < message.tensors.size(); i++) {
      message.tensors[i].sendShape(sock);
    }

    // send tensor data
    for (int i = 0; i < message.tensors.size(); i++) {
      message.tensors[i].sendData(sock);
    }

    // send metadata
    message.sendMetadata(sock);
    message.sendCommand(sock);

    // Receive header
    MessageHeader response_header;
    response_header.receiveFromSocket(sock);

    std::vector<TensorHeader> tensor_headers;
    for (int i = 0; i < response_header.n_tensors; i++) {
      TensorHeader tensor_header;
      tensor_header.receiveFromSocket(sock);
      tensor_headers.push_back(tensor_header);
      std::cout << "Receinvg header" << unsigned(tensor_header.rank)
                << std::endl;
    }

    std::vector<std::vector<int>> tensors_shapes;
    for (int i = 0; i < tensor_headers.size(); i++) {
      std::vector<int> tensor_shape;
      for (int j = 0; j < tensor_headers[i].rank; j++) {
        int shape_element;
        if (recv(sock, &shape_element, sizeof(int), 0) != sizeof(int))
          die("Receive Data Fails!");
        tensor_shape.push_back(shape_element);
      }
      tensors_shapes.push_back(tensor_shape);
      std::cout << "Receinvg Shape" << tensor_shape[0] << std::endl;
    }

    std::vector<TensorF> tensors;
    for (int i = 0; i < tensor_headers.size(); i++) {
      TensorF tensor;
      tensor.shape = tensors_shapes[i];
      tensor.receiveFromSocket(sock);
      tensors.push_back(tensor);
    }

    Message return_message;
    return_message.tensors = tensors;
    return_message.receiveMetadata(sock, response_header.metadata_size);
    return_message.receiveCommand(sock, response_header.trail_size);

    return return_message;
  }

  /**
   * Sends an opencv IMAGE through the socket
   */
  std::shared_ptr<uint8_t> sendAndReceiveData(float *array, int height,
                                              int width, int depth,
                                              int bytesPerElement,
                                              std::string command) {

    // Send headr
    int size = height * width * depth * bytesPerElement;

    nlohmann::json j = {{"currency", "USD"}};

    std::string nms = "gino";
    std::string last_string = j.dump();

    MessageHeader header(last_string.size(), 2, nms.size());
    sendHeader(sock, header);

    TensorHeader tensor_header;
    if (send(sock, &tensor_header, sizeof(TensorHeader), 0) !=
        sizeof(TensorHeader))
      die("Send Tensor Header fails!");
    if (send(sock, &tensor_header, sizeof(TensorHeader), 0) !=
        sizeof(TensorHeader))
      die("Send Tensor Header fails!");

    int image_size[3] = {256, 256, 3};
    if (send(sock, &image_size[0], sizeof(int) * 3, 0) != sizeof(int) * 3)
      die("Send Image Size fails!");

    int image_size2[3] = {1024, 448, 3};
    if (send(sock, &image_size2[0], sizeof(int) * 3, 0) != sizeof(int) * 3)
      die("Send Image Size fails!");

    uint8_t *data_1 = new uint8_t[256 * 256 * 3];
    uint8_t *data_2 = new uint8_t[1024 * 448 * 3];

    send(sock, &data_1[0], 256 * 256 * 3, 0);
    send(sock, &data_2[0], 1024 * 448 * 3, 0);

    if (send(sock, last_string.c_str(), last_string.size(), 0) !=
        last_string.size())
      die("Send Trail fails!");

    // Send Namespace
    if (send(sock, nms.c_str(), nms.size(), 0) != nms.size())
      die("Send Namespace fails!");

    // Receive header
    MessageHeader response_header;
    this->receiveHeader(response_header);
    std::cout << "Received header" << unsigned(response_header.metadata_size)
              << "," << unsigned(response_header.n_tensors) << ","
              << unsigned(response_header.trail_size) << std::endl;
    // response_header.print();

    // Receive raw float/bytes from server
    int expected_size = response_header.metadata_size;
    std::cout << "Expected size: " << expected_size << std::endl;
    std::shared_ptr<uint8_t> response_array(new uint8_t[expected_size],
                                            std::default_delete<uint8_t[]>());

    uint8_t *received_data = response_array.get();
    int received_size = 0;
    while (received_size < expected_size) {
      int remains = expected_size - received_size;
      int chunk_size = recv(sock, &(received_data)[received_size], remains, 0);
      received_size += chunk_size;
    }
    // conveert response array to string
    std::string response_string((char *)response_array.get(), expected_size);
    std::cout << "Received string: " << response_string << std::endl;
    // create json from string
    nlohmann::json response_json = nlohmann::json::parse(response_string);
    std::cout << "Received json: " << response_json << std::endl;
    return response_array;
  }
};

}; // namespace throwprotocol
