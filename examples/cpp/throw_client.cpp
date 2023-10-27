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

#include "Throw.hpp"

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

  throwprotocol::ThrowClientExample client(argv[1], atoi(argv[2]));

  float *data = new float[16]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

  throwprotocol::TensorF tensor;
  tensor.data = std::vector<float>(data, data + 16);
  tensor.shape = std::vector<int>{4, 4};

  std::vector<float> data_vector(data, data + 16);

  throwprotocol::Message message;
  message.metadata = nlohmann::json::parse("{\"command\":\"eye_matrix\"}");
  message.command = "pino";
  message.tensors.push_back(tensor);
  message.tensors.push_back(tensor);
  message.tensors.push_back(tensor);
  message.tensors.push_back(tensor);
  // std::cout << message.command.size() << std::endl;
  // std::cout << message.metadata.dump() << std::endl;

  while (true) {

    throwprotocol::Message returnMessage =
        client.sendAndReceiveMessage(message);
    std::cout << returnMessage.metadata.dump() << std::endl;
    // int height = 4;
    // int width = 4;
    // int depth = 1;
    // int bytes_per_element = 4;
    // std::shared_ptr<uint8_t> received_data = client.sendAndReceiveData(
    //     data, height, width, depth, bytes_per_element, "eye_matrix");
  }
}