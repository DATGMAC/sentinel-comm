cmake_minimum_required(VERSION 3.24)
project(sentinel_comm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Build for the GPU in this machine by default; override with
#   cmake -DCMAKE_CUDA_ARCHITECTURES="89;120" ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ── The bus + your handlers ──────────────────────────────────────────────────
# sentinel_comm.cu is the library; user_handlers.cu is the extension point —
# swap it for your own file defining sc_user_dispatch().
add_library(sentinel_comm STATIC
    src/sentinel_comm.cu
    src/user_handlers.cu
)
target_include_directories(sentinel_comm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sentinel_comm PUBLIC CUDA::cudart Threads::Threads)
set_target_properties(sentinel_comm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON      # required: sc_user_dispatch is device-linked
    POSITION_INDEPENDENT_CODE ON
)

# ── Example ──────────────────────────────────────────────────────────────────
add_executable(hello_roundtrip examples/hello_roundtrip.cpp)
target_link_libraries(hello_roundtrip sentinel_comm)
set_target_properties(hello_roundtrip PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ── Test ─────────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_roundtrip tests/test_roundtrip.cpp)
target_link_libraries(test_roundtrip sentinel_comm)
set_target_properties(test_roundtrip PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME roundtrip COMMAND test_roundtrip)

# ── Benchmark: dispatch latency vs kernel launch vs CUDA Graphs ─────────────
add_executable(bench_dispatch benchmarks/bench_dispatch.cu)
target_link_libraries(bench_dispatch sentinel_comm)
set_target_properties(bench_dispatch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
