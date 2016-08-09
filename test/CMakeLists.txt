set(CMAKE_CXX_FLAGS "-o3 -Wall -Wextra -std=c++14")

include_directories(../nn)

add_executable(t main.cpp)

target_link_libraries(t nn)
