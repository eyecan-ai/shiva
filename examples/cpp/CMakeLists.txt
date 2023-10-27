cmake_minimum_required(VERSION 2.8)
project(PostcardCppClient)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json 3.11.2 REQUIRED)
include_directories(include)

# find_package( OpenCV REQUIRED )

# add_executable( client_example client_example.cpp )
# target_link_libraries( client_example ${OpenCV_LIBS} )
add_executable(throw_client throw_client.cpp)

# target_link_libraries( client_exthrow_clientample ${OpenCV_LIBS} )