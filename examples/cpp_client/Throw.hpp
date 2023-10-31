#include "nlohmann/json.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <typeindex>
#include <typeinfo>
#include <unistd.h>
#include <unordered_map>
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
    this->CRC2 = (this->CRC + this->CRC) % 256;
  }

  void receiveFromSocket(int sock) {
    if (recv(sock, this, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Receive Data Fails!");
  }
};

struct TensorHeader {
  uint8_t rank = 3;
  uint8_t dtype = 3;

  void receive(int sock) {
    if (recv(sock, this, sizeof(TensorHeader), 0) != sizeof(TensorHeader))
      die("Receive Data Fails!");
  }
};

std::unordered_map<std::type_index, int8_t> TensorTypeMap = {
    {typeid(float), 1},      {typeid(double), 2},   {typeid(uint8_t), 3},
    {typeid(int8_t), 4},     {typeid(uint16_t), 5}, {typeid(int16_t), 6},
    {typeid(uint32_t), 7},   {typeid(int32_t), 8},  {typeid(uint64_t), 9},
    {typeid(int64_t), 10},   {typeid(double), 11},  {typeid(long double), 12},
    {typeid(long long), 13}, {typeid(bool), 17},
};

class BaseTensor {
public:
  std::vector<int> shape;
  std::type_index type;
  TensorHeader header;

  BaseTensor() : type(typeid(float)) {}
  virtual ~BaseTensor() = default;

  TensorHeader getHeader() {
    TensorHeader header;
    header.rank = this->shape.size();
    header.dtype = TensorTypeMap[type];
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

  void receiveShape(int sock) {
    std::vector<int> shape;
    for (int i = 0; i < this->header.rank; i++) {
      int shape_element;
      if (recv(sock, &shape_element, sizeof(int), 0) != sizeof(int))
        die("Receive Data Fails!");
      shape.push_back(shape_element);
    }
    this->shape = shape;
  }

  virtual void sendData(int sock) = 0;
  virtual void receiveData(int sock) = 0;
};
typedef std::shared_ptr<BaseTensor> BaseTensorPtr;

template <typename T> class Tensor : public BaseTensor {
public:
  std::vector<T> data;

  Tensor() : BaseTensor() { this->type = typeid(T); }
  ~Tensor() {}

  void debug() { std::cout << this->getHeader().dtype << std::endl; }

  void sendData(int sock) {
    if (send(sock, &this->data[0], sizeof(T) * this->data.size(), 0) !=
        sizeof(T) * this->data.size())
      die("Send Tensor Data fails!");
  }

  void receiveData(int sock) {

    int expected_size = 1;
    // exptected size is product of all shape elements * 4 bytes
    for (int i = 0; i < this->shape.size(); i++) {
      expected_size *= this->shape[i];
    }
    expected_size *= sizeof(T);

    std::shared_ptr<T> response_array(new T[expected_size],
                                      std::default_delete<T[]>());

    T *received_data = response_array.get();
    int received_size = 0;
    while (received_size < expected_size) {
      int remains = expected_size - received_size;
      int chunk_size = recv(sock, &(received_data)[received_size], remains, 0);
      received_size += chunk_size;
    }

    this->data = std::vector<T>(response_array.get(),
                                response_array.get() + expected_size);
  }
};

std::unordered_map<int8_t, std::type_index> TensorTypeMapInversed = {
    {1, typeid(Tensor<float>)},
};

std::variant<Tensor<float> *, Tensor<double> *> buildTensor(int dtype) {
  if (dtype == 1) {
    return new Tensor<float>();
  }

  else if (dtype == 2) {
    return new Tensor<double>();
  }

  //   else if (dtype == 3) {
  //     return Tensor<uint8_t>();
  //   } else if (dtype == 4) {
  //     return Tensor<int8_t>();
  //   } else if (dtype == 5) {
  //     return Tensor<uint16_t>();
  //   } else if (dtype == 6) {
  //     return Tensor<int16_t>();
  //   } else if (dtype == 7) {
  //     return Tensor<uint32_t>();
  //   } else if (dtype == 8) {
  //     return Tensor<int32_t>();
  //   } else if (dtype == 9) {
  //     return Tensor<uint64_t>();
  //   } else if (dtype == 10) {
  //     return Tensor<int64_t>();
  //   } else if (dtype == 11) {
  //     return Tensor<double>();
  //   } else if (dtype == 12) {
  //     return Tensor<long double>();

  //   } else if (dtype == 13) {
  //     return Tensor<long long>();
  //   } else if (dtype == 17) {
  //     return Tensor<bool>();
  //   }

  else {
    return new Tensor<float>();
  }
}
// create a typedef which is a variant of Tensor<float> and Tensor<uint8_t>
typedef std::variant<Tensor<float>, Tensor<uint8_t>> TensorVariant;

class Message {
public:
  nlohmann::json metadata;
  std::string command;
  std::vector<std::shared_ptr<BaseTensor>> tensors;
  Message() : metadata(nlohmann::json::object()), command(""), tensors() {}

  MessageHeader buildHeader() {
    MessageHeader header(this->metadata.dump().size(), this->tensors.size(),
                         this->command.size());
    return header;
  }

  MessageHeader receiveHeader(int sock) {
    MessageHeader header;
    if (recv(sock, &header, sizeof(MessageHeader), 0) != sizeof(MessageHeader))
      die("Receive Data Fails!");
    return header;
  }

  TensorHeader receiveTensorHeader(int sock) {
    TensorHeader header;
    if (recv(sock, &header, sizeof(TensorHeader), 0) != sizeof(TensorHeader))
      die("Receive Data Fails!");
    return header;
  }

  std::vector<int> receiveTensorShape(int sock, TensorHeader &th) {
    std::vector<int> shape;
    for (int i = 0; i < th.rank; i++) {
      int shape_element;
      if (recv(sock, &shape_element, sizeof(int), 0) != sizeof(int))
        die("Receive Data Fails!");
      shape.push_back(shape_element);
    }
    return shape;
  }

  BaseTensorPtr receiveTensor(int sock, const TensorHeader &th,
                              const std::vector<int> &shape) {

    BaseTensorPtr tensor;
    if (th.dtype == 1) {
      tensor = std::make_shared<Tensor<float>>();
      tensor->shape = shape;
      tensor->receiveData(sock);
    } else if (th.dtype == 3) {
      tensor = std::make_shared<Tensor<uint8_t>>();
      tensor->shape = shape;
      tensor->receiveData(sock);
    } else {
      throw std::runtime_error("Unknown dtype " + std::to_string(th.dtype));
    }
    return tensor;
  }

  static Message receive(int sock) {
    Message returnMessage;
    MessageHeader returnHeader = returnMessage.receiveHeader(sock);

    for (int i = 0; i < returnHeader.n_tensors; i++) {
      TensorHeader th = returnMessage.receiveTensorHeader(sock);
      std::vector<int> shape = returnMessage.receiveTensorShape(sock, th);
      BaseTensorPtr tensor = returnMessage.receiveTensor(sock, th, shape);
    }
    returnMessage.receiveMetadata(sock, returnHeader.metadata_size);
    returnMessage.receiveCommand(sock, returnHeader.trail_size);
    return returnMessage;
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
    MessageHeader header = this->buildHeader();
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

  void sendMessage(int sock) {
    this->sendHeader(sock);
    for (int i = 0; i < this->tensors.size(); i++) {
      this->tensors[i]->sendHeader(sock);
      this->tensors[i]->sendShape(sock);
      this->tensors[i]->sendData(sock);
    }
    this->sendMetadata(sock);
    this->sendCommand(sock);
  }
};

class ThrowClientExample {

public:
  /* Socket parameters */
  int sock;                         /* socket handle */
  struct sockaddr_in echoServAddr;  /* server address */
  unsigned short serverPort = 8000; /* server port */
  std::string server_address;       /* Server IP address (dotted quad) */
  unsigned short port;              /* Input Image */

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

    message.sendMessage(this->sock);
    return Message::receive(this->sock);
  }
};
}; // namespace throwprotocol
