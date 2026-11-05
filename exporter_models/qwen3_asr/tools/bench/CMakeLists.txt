# qwen3_asr_bench: end-to-end Qwen3-ASR .pte runner (see main.cpp).
#
# Copy this directory to <executorch>/examples/models/qwen3_asr_bench and build
# it against an installed ExecuTorch, like examples/models/llama:
#
#   cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
#     -DCMAKE_INSTALL_PREFIX=cmake-out-android -DCMAKE_BUILD_TYPE=Release \
#     -DPYTHON_EXECUTABLE=python -DEXECUTORCH_BUILD_XNNPACK=ON \
#     -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
#     -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
#     -Bcmake-out-android/examples/models/qwen3_asr_bench \
#     examples/models/qwen3_asr_bench
cmake_minimum_required(VERSION 3.24)
project(qwen3_asr_bench)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries
    executorch extension_module_static extension_data_loader extension_tensor
    extension_flat_tensor extension_named_data_map extension_threadpool
    pthreadpool cpuinfo
)

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib optimized_kernels
       portable_kernels cpublas eigen_blas
  )
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

# quantized_decomposed::embedding_byte for the XNNPACK token embedding
list(APPEND link_libraries quantized_kernels quantized_ops_lib)
executorch_target_link_options_shared_lib(quantized_ops_lib)

# llama::update_cache / llama::custom_sdpa for the XNNPACK decoder
list(APPEND link_libraries custom_ops)
executorch_target_link_options_shared_lib(custom_ops)

list(APPEND link_libraries xnnpack_backend XNNPACK xnnpack-microkernels-prod)
if(TARGET kleidiai)
  list(APPEND link_libraries kleidiai)
endif()
executorch_target_link_options_shared_lib(xnnpack_backend)

if(TARGET vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
endif()

if(ANDROID)
  list(APPEND link_libraries log)
endif()

add_executable(qwen3_asr_bench main.cpp)
target_include_directories(
  qwen3_asr_bench
  PUBLIC ${EXECUTORCH_ROOT}/..
         ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
         ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
)
target_link_libraries(qwen3_asr_bench PUBLIC ${link_libraries})
target_compile_options(qwen3_asr_bench PUBLIC -Wno-deprecated-declarations -fPIC)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_link_options_gc_sections(qwen3_asr_bench)
  target_link_options(qwen3_asr_bench PRIVATE "LINKER:-s")
endif()
