cmake_minimum_required(VERSION 3.10)

project(clfft_test)

set(CMAKE_CXX_STANDARD 11)


find_package(OpenCL REQUIRED)

# 确保 clFFT 的路径是正确的
set(CLFFT_DIR "~/codegram/package/clFFT/src/build/library")  # 替换为你的实际路径

# 包含 clFFT 的头文件
include_directories("~/codegram/package/clFFT/src/build/include")  # 替换为实际的头文件路径

add_executable(clfft_test ts.cpp timeKeeper.cpp)

target_link_libraries(clfft_test PRIVATE OpenCL::OpenCL ${CLFFT_DIR}/libclFFT.so)

