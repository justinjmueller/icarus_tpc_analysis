cmake_minimum_required(VERSION 3.10)

project(generator)

include_directories(${ROOT_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(generator generator.cc)