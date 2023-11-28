cmake_minimum_required(VERSION 3.26)
project(mpi)

set(CMAKE_CXX_STANDARD 17)

#add_executable(mpi main.cpp)
#include_directories("/opt/homebrew/Cellar/openssl@3/3.2.0/include")
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})
add_executable(mpi md5.cpp)
#target_link_libraries(mpi "/opt/homebrew/Cellar/openssl@3/3.2.0/lib/libssl.dylib")
target_link_libraries(mpi ${OPENSSL_LIBRARIES})