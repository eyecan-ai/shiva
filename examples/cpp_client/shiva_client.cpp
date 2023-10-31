#include "Shiva.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono;
using namespace shiva;

void die(std::string errorMessage) {
  std::cout << errorMessage << std::endl;
  exit(1);
}

template <typename DataType>
std::shared_ptr<Tensor<DataType>> createTensor(std::vector<int> shape,
                                               int fill_value = 0) {
  int total_size = 1;
  std::for_each(shape.begin(), shape.end(), [&](int n) { total_size *= n; });
  DataType *data = new DataType[total_size];

  std::shared_ptr<Tensor<DataType>> tensor =
      std::make_shared<Tensor<DataType>>();

  std::fill_n(data, total_size, fill_value);

  tensor->data = std::vector<DataType>(data, data + total_size);
  tensor->shape = shape;
  return tensor;
}

int main(int argc, char *argv[]) {

  /* Socket parameters */
  int sock;                         /* Socket descriptor */
  struct sockaddr_in echoServAddr;  /* Echo server address */
  unsigned short serverPort = 6174; /* Echo server port */

  if (argc < 3) {
    std::cout << "Not enought arguments..." << std::endl;
    exit(1);
  }

  // Create Shiva Client
  ShivaClientExample client(argv[1], atoi(argv[2]));

  // Create 3 random tensors
  Tensor<uint8_t>::Ptr tensor_1 = createTensor<uint8_t>({1920, 1080, 3});
  Tensor<uint8_t>::Ptr tensor_2 = createTensor<uint8_t>({1920, 1080, 3});
  Tensor<float>::Ptr tensor_3 = createTensor<float>({1});
  Tensor<uint32_t>::Ptr tensor_4 = createTensor<uint32_t>({10, 10});

  // Empty shiva message
  ShivaMessage message;

  // Populate metadata with json
  message.metadata = {{"counter", 0},
                      {"__tensors__",
                       {
                           "tensor_1",
                           "tensor_2",
                           "counter",
                           "tensor_4",
                       }}};

  // Set command (aka namespace)
  message.command = "inference";

  // Add tensors to message
  message.tensors.push_back(tensor_1);
  message.tensors.push_back(tensor_2);
  message.tensors.push_back(tensor_3);
  message.tensors.push_back(tensor_4);

  while (true) {

    auto start = high_resolution_clock::now();

    // Sending/Grab message
    ShivaMessage returnMessage = client.sendAndReceiveMessage(message);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "FPS: " << 1000000 / duration.count() << std::endl;
    std::cout << "metadata: " << returnMessage.metadata.dump() << std::endl;

    message = returnMessage;
  }
}