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

add_library(rr_kernels src/transport_kernels.cpp)
target_include_directories(rr_kernels PUBLIC include)
target_link_libraries(rr_kernels PUBLIC OpenMP::OpenMP_CXX Threads::Threads)

add_executable(rr_benchmark
    src/main.cpp
)

target_link_libraries(rr_benchmark PRIVATE
    rr_kernels
)

# Automated testing
enable_testing()
add_executable(rr_kernel_tests tests/kernel_tests.cpp)
target_link_libraries(rr_kernel_tests PRIVATE rr_kernels)
add_test(NAME kernel_verification COMMAND rr_kernel_tests)
add_test(NAME smoke_test COMMAND rr_benchmark --smoke --threads 2)
add_test(NAME smoke_p0 COMMAND rr_benchmark --smoke --threads 2 --moments 1)
add_test(NAME smoke_max_groups COMMAND rr_benchmark --smoke --threads 2 --groups 256)

function(add_rejected_input_test name)
    add_test(NAME ${name} COMMAND rr_benchmark ${ARGN})
    set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

add_rejected_input_test(reject_zero_groups --groups 0)
add_rejected_input_test(reject_negative_groups --groups -1)
add_rejected_input_test(reject_excess_groups --groups 257)
add_rejected_input_test(reject_invalid_groups --groups abc)
add_rejected_input_test(reject_partial_integer --groups 47junk)
add_rejected_input_test(reject_integer_overflow --groups 999999999999999999999999)
add_rejected_input_test(reject_zero_moments --moments 0)
add_rejected_input_test(reject_dimension_overflow --moments 9223372036854775807)
add_rejected_input_test(reject_zero_threads --threads 0)
add_rejected_input_test(reject_thread_overflow --threads 2147483648)
add_rejected_input_test(reject_zero_segments --segments 0)
add_rejected_input_test(reject_zero_trials --trials 0)
add_rejected_input_test(reject_missing_value --groups)
add_rejected_input_test(reject_unknown_option --unknown)

get_property(benchmark_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${benchmark_tests} PROPERTIES TIMEOUT 45)
