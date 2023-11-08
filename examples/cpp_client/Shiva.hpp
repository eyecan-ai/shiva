#include "nlohmann/json.hpp"
#include <arpa/inet.h>
#include <cstdint>
#include <iostream>
#include <netinet/tcp.h>
#include <string>
#include <typeindex>
#include <unistd.h>
#include <unordered_map>

namespace shiva {

/**
 * Bigendian uint32_t
 */
class be_uint32_t {
public:
  be_uint32_t() : be_val_(0) {}
  be_uint32_t(const uint32_t &val) : be_val_(htonl(val)) {}
  operator uint32_t() const { return ntohl(be_val_); }

private:
  uint32_t be_val_;
} __attribute__((packed));

/**
 * DIEs
 */
void die(std::string errorMessage) {
  std::cout << errorMessage << std::endl;
  exit(1);
}

std::unordered_map<std::type_index, int8_t> TensorTypeMap = {
    {typeid(float), 1},      {typeid(double), 2},   {typeid(uint8_t), 3},
    {typeid(int8_t), 4},     {typeid(uint16_t), 5}, {typeid(int16_t), 6},
    {typeid(uint32_t), 7},   {typeid(int32_t), 8},  {typeid(uint64_t), 9},
    {typeid(int64_t), 10},   {typeid(double), 11},  {typeid(long double), 12},
    {typeid(long long), 13}, {typeid(bool), 17},
};

struct MessageHeader {
  uint8_t MAGIC[4];
  be_uint32_t metadata_size;
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

    // compute crc summing all
    this->CRC = 0;
    this->CRC +=
        this->MAGIC[0] + this->MAGIC[1] + this->MAGIC[2] + this->MAGIC[3];
    this->CRC += this->metadata_size;
    this->CRC += this->n_tensors;
    this->CRC += this->trail_size;
    this->CRC = this->CRC % 256;

    // compute crc2 summing all previous
    this->CRC2 = (this->CRC + this->CRC) % 256;
  }
};

struct TensorHeader {
  uint8_t rank = 0;
  uint8_t dtype = 0;

  void receive(int sock) {
    if (recv(sock, this, sizeof(TensorHeader), 0) != sizeof(TensorHeader))
      die("Receive Data Fails!");
  }
};

class BaseTensor {
public:
  std::vector<uint32_t> shape;
  std::type_index type;
  TensorHeader header;

  BaseTensor() : type(typeid(float)) {}
  virtual ~BaseTensor() = default;

  TensorHeader buildHeader() {
    TensorHeader header;
    header.rank = this->shape.size();
    header.dtype = TensorTypeMap[type];
    return header;
  }

  void sendHeader(int sock) {
    TensorHeader header = this->buildHeader();
    if (send(sock, &header, sizeof(TensorHeader), 0) != sizeof(TensorHeader))
      die("Send Tensor Header fails!");
  }

  void sendShape(int sock) {

    std::vector<be_uint32_t> beshape =
        std::vector<be_uint32_t>(this->shape.begin(), this->shape.end());

    if (send(sock, &beshape[0], sizeof(uint32_t) * beshape.size(), 0) !=
        sizeof(uint32_t) * beshape.size())
      die("Send Tensor Size fails!");
  }

  // void receiveShape(int sock) {
  //   std::vector<uint32_t> shape;
  //   for (int i = 0; i < this->header.rank; i++) {
  //     be_uint32_t shape_element;
  //     if (recv(sock, &shape_element, sizeof(be_uint32_t), 0) !=
  //         sizeof(be_uint32_t))
  //       die("Receive Data Fails!");
  //     shape.push_back(shape_element);
  //   }
  //   this->shape = shape;
  // }

  virtual void copyData(void *data) = 0;
  virtual void sendData(int sock) = 0;
  virtual void receiveData(int sock) = 0;
};
typedef std::shared_ptr<BaseTensor> BaseTensorPtr;

template <typename T> class Tensor : public BaseTensor {
public:
  std::vector<T> data;
  typedef std::shared_ptr<Tensor<T>> Ptr;

  Tensor() : BaseTensor() { this->type = typeid(T); }
  ~Tensor() {}

  void sendData(int sock) {
    // print type index
    if (send(sock, &this->data[0], sizeof(T) * this->data.size(), 0) !=
        sizeof(T) * this->data.size())
      die("Send Tensor Data fails!");
  }

  void copyData(void *data) {
    std::copy_n(this->data.begin(), this->data.size(), (T *)data);
  }

  void receiveData(int sock) {

    int elements = 1;
    // exptected size is product of all shape elements * 4 bytes
    for (int i = 0; i < this->shape.size(); i++) {
      elements *= this->shape[i];
    }
    int expected_size = elements * sizeof(T);

    std::shared_ptr<T> response_array(new T[expected_size],
                                      std::default_delete<T[]>());

    T *received_data = response_array.get();
    int received_size = 0;
    while (received_size < expected_size) {
      int remains = expected_size - received_size;
      int chunk_size = recv(sock, &(received_data)[received_size], remains, 0);
      received_size += chunk_size;
    }

    this->data.clear();
    this->data = std::vector<T>(elements);
    std::copy_n(received_data, elements, this->data.begin());
  }
};

std::unordered_map<int8_t, std::type_index> TensorTypeMapInversed = {
    {1, typeid(Tensor<float>)},
};

class ShivaMessage {

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

  std::vector<uint32_t> receiveTensorShape(int sock, TensorHeader &th) {

    std::vector<be_uint32_t> beshape(th.rank);
    if (recv(sock, &beshape[0], sizeof(be_uint32_t) * th.rank, 0) !=
        sizeof(be_uint32_t) * th.rank)
      die("Receive Data Fails!");
    std::vector<uint32_t> shape(beshape.begin(), beshape.end());
    return shape;
  }

  BaseTensorPtr receiveTensor(int sock, const TensorHeader &th,
                              const std::vector<uint32_t> &shape) {

    BaseTensorPtr tensor;
    if (th.dtype == 1) {
      tensor = std::make_shared<Tensor<float>>();
      tensor->header = th;
      tensor->shape = shape;
      tensor->receiveData(sock);
    } else if (th.dtype == 3) {
      tensor = std::make_shared<Tensor<uint8_t>>();
      tensor->header = th;
      tensor->shape = shape;
      tensor->receiveData(sock);
    } else if (th.dtype == 7) {
      tensor = std::make_shared<Tensor<uint32_t>>();
      tensor->header = th;
      tensor->shape = shape;
      tensor->receiveData(sock);
    } else {
      throw std::runtime_error("Not implemented dtype " +
                               std::to_string(th.dtype));
    }
    return tensor;
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
    if (send(sock, this->metadata.dump().c_str(), this->metadata.dump().size(),
             0) != this->metadata.dump().size())
      die("Send Metadata fails!");
  }

  void sendCommand(int sock) {
    if (send(sock, this->command.c_str(), this->command.size(), 0) !=
        this->command.size())
      die("Send Command fails!");
  }

public:
  MessageHeader buildHeader() {
    MessageHeader header(this->metadata.dump().size(), this->tensors.size(),
                         this->command.size());
    return header;
  }
  nlohmann::json metadata;
  std::string command;
  std::vector<std::shared_ptr<BaseTensor>> tensors;
  ShivaMessage() : metadata(nlohmann::json::object()), command(""), tensors() {}
  // create copy constructor
  ShivaMessage(const ShivaMessage &other) {
    this->metadata = other.metadata;
    this->command = other.command;
    this->tensors = other.tensors;
  }

  static ShivaMessage receive(int sock) {
    ShivaMessage returnMessage;
    MessageHeader returnHeader = returnMessage.receiveHeader(sock);

    for (int i = 0; i < returnHeader.n_tensors; i++) {
      TensorHeader th = returnMessage.receiveTensorHeader(sock);
      std::cout << "HERE\n";
      std::vector<uint32_t> shape = returnMessage.receiveTensorShape(sock, th);

      for (int i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << " ";
      }

      BaseTensorPtr tensor = returnMessage.receiveTensor(sock, th, shape);
      tensor->header = th;
      tensor->shape = shape;
      returnMessage.tensors.push_back(tensor);
    }
    returnMessage.receiveMetadata(sock, returnHeader.metadata_size);
    returnMessage.receiveCommand(sock, returnHeader.trail_size);
    return returnMessage;
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

class ShivaClientExample {

public:
  /* Socket parameters */
  int sock;                         /* socket handle */
  struct sockaddr_in echoServAddr;  /* server address */
  unsigned short serverPort = 8000; /* server port */
  std::string server_address;       /* Server IP address (dotted quad) */
  unsigned short port;              /* Input Image */

  ShivaClientExample(std::string server_address, unsigned short port) {

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

  ShivaMessage sendAndReceiveMessage(ShivaMessage &message) {
    message.sendMessage(this->sock);
    return ShivaMessage::receive(this->sock);
  }
};
}; // namespace shiva
