cmake_minimum_required(VERSION 2.8)

project(different)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -stdlib=libstdc++ -Wall")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest/googletest/build
)

add_executable(test_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/test/testDifferent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/runTests.cpp
)

target_link_libraries(test_runner
    glog
    gtest
    pthread
)
