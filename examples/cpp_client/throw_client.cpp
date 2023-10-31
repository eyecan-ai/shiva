#include "Throw.hpp"
#include <algorithm>
#include <arpa/inet.h> /* for sockaddr_in and inet_addr() */
#include <chrono>
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

using namespace std::chrono;

void die(std::string errorMessage) {
  std::cout << errorMessage << std::endl;
  exit(1);
}

int main(int argc, char *argv[]) {

  /* Socket parameters */
  int sock;                         /* Socket descriptor */
  struct sockaddr_in echoServAddr;  /* Echo server address */
  unsigned short serverPort = 8000; /* Echo server port */

  if (argc < 3) {
    std::cout << "Not enought arguments..." << std::endl;
    exit(1);
  }
  using namespace throwprotocol;
  // throwprotocol::Tensor<float> tensor;
  // throwprotocol::Tensor<int> tensor2;

  // std::variant<Tensor<float>, Tensor<double>> tensorX = buildTensor(1);
  // std::variant<Tensor<float>, Tensor<double>> tensorY = buildTensor(2);

  // std::type_info t = TensorTypeMapInversed[1];
  // std::cout << throwprotocol::TensorTypeMapInversed[1] << std::endl;

  throwprotocol::ThrowClientExample client(argv[1], atoi(argv[2]));

  float *data = new float[16]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

  std::shared_ptr<Tensor<float>> tensor = std::make_shared<Tensor<float>>();
  tensor->data = std::vector<float>(data, data + 16);
  tensor->shape = std::vector<int>{4, 4};

  int width = 1920;
  int height = 1080;
  int channels = 3;
  int total_size = width * height * channels;
  uint8_t *datauint8 = new uint8_t[total_size]{0};
  std::shared_ptr<Tensor<uint8_t>> tensor2 =
      std::make_shared<Tensor<uint8_t>>();
  tensor2->data = std::vector<uint8_t>(datauint8, datauint8 + total_size);
  tensor2->shape = std::vector<int>{height, width, channels};

  std::vector<std::shared_ptr<BaseTensor>> tensors;
  // tensors.push_back(tensor);
  tensors.push_back(tensor2);
  tensors.push_back(tensor2);
  tensors.push_back(tensor2);
  tensors.push_back(tensor2);
  tensors.push_back(tensor2);
  // tensors.push_back(tensor2);

  // std::cout << "T" << unsigned(tensors[0]->data.size()) << std::endl;
  // std::cout << "T" << unsigned(tensors[1]->shape.size()) << std::endl;

  Message message;
  int a = 1;
  message.metadata = nlohmann::json::parse("{\"command\":\"eye_matrix\"}");
  message.command = "pino";
  message.tensors = tensors;

  while (true) {

    // measure execution time of the followin line
    auto start = high_resolution_clock::now();
    Message returnMessage = client.sendAndReceiveMessage(message);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "FPS: " << 1000000 / duration.count() << std::endl;
  }
}