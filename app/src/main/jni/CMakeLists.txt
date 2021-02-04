project(vkpeakncnn)

cmake_minimum_required(VERSION 3.4.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210203-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(vkpeakncnn SHARED vkpeakncnn_jni.cpp)

target_link_libraries(vkpeakncnn ncnn)
