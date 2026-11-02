cmake_minimum_required(VERSION 3.14)
project(random_ray_loop_reorder_and_locks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# Find OpenMP and Threads
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

include_directories(include)

add_executable(rr_benchmark
    src/main.cpp
    src/transport_kernels.cpp
)

target_link_libraries(rr_benchmark PRIVATE
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Automated testing
enable_testing()
add_test(NAME smoke_test COMMAND rr_benchmark --smoke)
