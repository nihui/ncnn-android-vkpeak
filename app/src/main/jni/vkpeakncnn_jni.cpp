// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/log.h>

#include <sys/system_properties.h>

#include <jni.h>

#include <dlfcn.h>
#include <float.h>
#include <string>
#include <vector>

// ncnn
#include <benchmark.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

// Driver indices match MainActivity's selector.
static bool g_vulkan_ready = false;
static int g_vulkan_type = 0;
static bool g_turnip_supported = false;
static bool g_momoten_available = false;

static ncnn::VulkanDevice* get_vulkan_device()
{
    // ncnn retains its previous GPU count after a failed instance creation.
    return g_vulkan_ready ? ncnn::get_gpu_device() : 0;
}

static bool is_momoten(const ncnn::GpuInfo& info)
{
    return strcmp(info.driver_name(), "momoten") == 0
        || strncmp(info.device_name(), "momoten: ", 9) == 0;
}

static bool create_vulkan(int type)
{
    const char* driver_path = 0;
    if (type == 1)
        driver_path = "libvulkan_freedreno.so";
    if (type == 2)
        driver_path = "libmomoten.so";

    g_vulkan_ready = false;
    g_vulkan_type = 0;
    if (ncnn::create_gpu_instance(driver_path) != 0 || ncnn::get_gpu_count() == 0)
        return false;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev || !vkdev->is_valid())
        return false;

    g_vulkan_ready = true;
    if (is_momoten(vkdev->info))
        g_vulkan_type = 2;
    else if (vkdev->info.driver_id() == VK_DRIVER_ID_MESA_TURNIP)
        g_vulkan_type = 1;

    // simplevk can silently fall back to another driver, even for an explicit path.
    return type == 0 || g_vulkan_type == type;
}

#define REPEAT_1(...) #__VA_ARGS__
#define REPEAT_2(...) REPEAT_1(__VA_ARGS__) REPEAT_1(__VA_ARGS__)
#define REPEAT_4(...) REPEAT_2(__VA_ARGS__) REPEAT_2(__VA_ARGS__)
#define REPEAT_8(...) REPEAT_4(__VA_ARGS__) REPEAT_4(__VA_ARGS__)
#define REPEAT_16(...) REPEAT_8(__VA_ARGS__) REPEAT_8(__VA_ARGS__)

static const char glsl_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    afp c = afp(gx);

    afp a = c;
    afp b = afp(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = float(c);
}
)";

static const char glsl_p1_dual_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    afp c0 = afp(gx);
    afp c1 = afp(lx);

    afp a = c0;
    afp b = c1;

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = float(c0);
}
)";

static const char glsl_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    afpvec4 c = afpvec4(gx);

    afpvec4 a = c + afpvec4(0,1,2,-3);
    afpvec4 b = afpvec4(lx) + afpvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = float((c[0] + c[1]) + (c[2] + c[3]));
}
)";

static const char glsl_p4_dual_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    afpvec4 c0 = afpvec4(gx);
    afpvec4 c1 = afpvec4(lx);

    afpvec4 a = c0 + afpvec4(0,1,2,-3);
    afpvec4 b = c1 + afpvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = float((c0[0] + c0[1]) + (c0[2] + c0[3]));
}
)";

static const char glsl_fp64_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    double c = double(gx);

    double a = c;
    double b = double(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = c;
}
)";

static const char glsl_fp64_p1_dual_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    double c0 = double(gx);
    double c1 = double(lx);

    double a = c0;
    double b = c1;

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_fp64_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    dvec4 c = dvec4(gx);

    dvec4 a = c + dvec4(0,1,2,-3);
    dvec4 b = dvec4(lx) + dvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = (c[0] + c[1]) + (c[2] + c[3]);
}
)";

static const char glsl_fp64_p4_dual_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    dvec4 c0 = dvec4(gx);
    dvec4 c1 = dvec4(lx);

    dvec4 a = c0 + dvec4(0,1,2,-3);
    dvec4 b = c1 + dvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = (c0[0] + c0[1]) + (c0[2] + c0[3]);
}
)";

static const char glsl_int32_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int c = int(gx);

    int a = c;
    int b = int(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = c;
}
)";

static const char glsl_int32_p1_dual_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int c0 = int(gx);
    int c1 = int(lx);

    int a = c0;
    int b = c1;

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_int32_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    ivec4 c = ivec4(gx);

    ivec4 a = c + ivec4(0,1,2,-3);
    ivec4 b = ivec4(lx) + ivec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = (c[0] + c[1]) + (c[2] + c[3]);
}
)";

static const char glsl_int32_p4_dual_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    ivec4 c0 = ivec4(gx);
    ivec4 c1 = ivec4(lx);

    ivec4 a = c0 + ivec4(0,1,2,-3);
    ivec4 b = c1 + ivec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = (c0[0] + c0[1]) + (c0[2] + c0[3]);
}
)";

static const char glsl_int16_p1_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int16_t c = int16_t(gx);

    int16_t a = c;
    int16_t b = int16_t(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = int(c);
}
)";

static const char glsl_int16_p1_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int16_t c0 = int16_t(gx);
    int16_t c1 = int16_t(lx);

    int16_t a = c0;
    int16_t b = c1;

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = int(c0);
}
)";

static const char glsl_int16_p4_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    i16vec4 c = i16vec4(gx);

    i16vec4 a = c + i16vec4(0,1,2,-3);
    i16vec4 b = i16vec4(lx) + i16vec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = int((c[0] + c[1]) + (c[2] + c[3]));
}
)";

static const char glsl_int16_p4_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    i16vec4 c0 = i16vec4(gx);
    i16vec4 c1 = i16vec4(lx);

    i16vec4 a = c0 + i16vec4(0,1,2,-3);
    i16vec4 b = c1 + i16vec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = int((c0[0] + c0[1]) + (c0[2] + c0[3]));
}
)";

static const char glsl_int64_p1_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int64_t c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int64_t c = int64_t(gx);

    int64_t a = c;
    int64_t b = int64_t(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = c;
}
)";

static const char glsl_int64_p1_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int64_t c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int64_t c0 = int64_t(gx);
    int64_t c1 = int64_t(lx);

    int64_t a = c0;
    int64_t b = c1;

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_int64_p4_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int64_t c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    i64vec4 c = i64vec4(gx);

    i64vec4 a = c + i64vec4(0,1,2,-3);
    i64vec4 b = i64vec4(lx) + i64vec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = a * c + b;)
    R"(}

    c_blob_data[gx] = (c[0] + c[1]) + (c[2] + c[3]);
}
)";

static const char glsl_int64_p4_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int64_t c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    i64vec4 c0 = i64vec4(gx);
    i64vec4 c1 = i64vec4(lx);

    i64vec4 a = c0 + i64vec4(0,1,2,-3);
    i64vec4 b = c1 + i64vec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = a * c0 + b; c1 = a * c1 + b;)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = (c0[0] + c0[1]) + (c0[2] + c0[3]);
}
)";

static const char glsl_int8_p4_data[] = R"(
#version 450

#extension GL_EXT_integer_dot_product: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int c = int(gx);

    int a = int(gx);
    int b = int(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = dotPacked4x8AccSatEXT(a, b, c);)
    R"(}

    c_blob_data[gx] = c;
}
)";

static const char glsl_int8_p4_dual_data[] = R"(
#version 450

#extension GL_EXT_integer_dot_product: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int c0 = int(gx);
    int c1 = int(lx);

    int a = int(gx);
    int b = int(lx);

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = dotPacked4x8AccSatEXT(a, b, c0); c1 = dotPacked4x8AccSatEXT(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_bf16_p4_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_bfloat16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    bfloat16_t c = bfloat16_t(gx);

    u16vec4 a = uint16_t(gx) + u16vec4(0,1,2,3);
    bf16vec4 b = uintBitsToBFloat16EXT(uint16_t(lx) + u16vec4(2,3,5,7));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_4(c = dot(uintBitsToBFloat16EXT(a), b); a.x = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b); a.y = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b); a.z = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b); a.w = bfloat16BitsToUintEXT(c);)
    R"(}

    c_blob_data[gx] = float(c);
}
)";

static const char glsl_bf16_p4_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_bfloat16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    bfloat16_t c0 = bfloat16_t(gx);
    bfloat16_t c1 = bfloat16_t(lx);

    u16vec4 a0 = uint16_t(gx) + u16vec4(0,1,2,3);
    u16vec4 a1 = uint16_t(gx) + u16vec4(10,21,32,43);
    bf16vec4 b = uintBitsToBFloat16EXT(uint16_t(lx) + u16vec4(2,3,5,7));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_4(c0 = dot(uintBitsToBFloat16EXT(a0), b); a0.x = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b); a1.y = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b); a0.z = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b); a1.w = bfloat16BitsToUintEXT(c1);)
    R"(}

    c_blob_data[gx] = float(c0) + float(c1);
}
)";

static const char glsl_fp16_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#if ncnn_VK_KHR_cooperative_matrix
#extension GL_KHR_cooperative_matrix: require
#elif ncnn_VK_NV_cooperative_matrix
#extension GL_NV_cooperative_matrix: require
#endif

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

#if ncnn_VK_KHR_cooperative_matrix
    coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
#elif ncnn_VK_NV_cooperative_matrix
    fcoopmatNV<16, SCOPE, M, K> a = fcoopmatNV<16, SCOPE, M, K>(float(gx));
    fcoopmatNV<16, SCOPE, K, N> b = fcoopmatNV<16, SCOPE, K, N>(float(lx));

    fcoopmatNV<16, SCOPE, M, N> c = fcoopmatNV<16, SCOPE, M, N>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAddNV(a, b, c);)
    R"(}

    coopMatStoreNV(c, c_blob_data, gx * (M * N) / 2, N / 2, false);
#endif
}
)";

static const char glsl_fp16_matrix_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#if ncnn_VK_KHR_cooperative_matrix
#extension GL_KHR_cooperative_matrix: require
#elif ncnn_VK_NV_cooperative_matrix
#extension GL_NV_cooperative_matrix: require
#endif

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

#if ncnn_VK_KHR_cooperative_matrix
    coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
#elif ncnn_VK_NV_cooperative_matrix
    fcoopmatNV<16, SCOPE, M, K> a = fcoopmatNV<16, SCOPE, M, K>(float(gx));
    fcoopmatNV<16, SCOPE, K, N> b = fcoopmatNV<16, SCOPE, K, N>(float(lx));

    fcoopmatNV<16, SCOPE, M, N> c0 = fcoopmatNV<16, SCOPE, M, N>(float(gx));
    fcoopmatNV<16, SCOPE, M, N> c1 = fcoopmatNV<16, SCOPE, M, N>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAddNV(a, b, c0); c1 = coopMatMulAddNV(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx * (M * N) / 2, N / 2, false);
#endif
}
)";

static const char glsl_fp16_fp32_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#if ncnn_VK_KHR_cooperative_matrix
#extension GL_KHR_cooperative_matrix: require
#elif ncnn_VK_NV_cooperative_matrix
#extension GL_NV_cooperative_matrix: require
#endif

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

#if ncnn_VK_KHR_cooperative_matrix
    coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
#elif ncnn_VK_NV_cooperative_matrix
    fcoopmatNV<16, SCOPE, M, K> a = fcoopmatNV<16, SCOPE, M, K>(float(gx));
    fcoopmatNV<16, SCOPE, K, N> b = fcoopmatNV<16, SCOPE, K, N>(float(lx));

    fcoopmatNV<32, SCOPE, M, N> c = fcoopmatNV<32, SCOPE, M, N>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAddNV(a, b, c);)
    R"(}

    coopMatStoreNV(c, c_blob_data, gx * (M * N), N, false);
#endif
}
)";

static const char glsl_fp16_fp32_matrix_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#if ncnn_VK_KHR_cooperative_matrix
#extension GL_KHR_cooperative_matrix: require
#elif ncnn_VK_NV_cooperative_matrix
#extension GL_NV_cooperative_matrix: require
#endif

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

#if ncnn_VK_KHR_cooperative_matrix
    coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<float16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<float16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
#elif ncnn_VK_NV_cooperative_matrix
    fcoopmatNV<16, SCOPE, M, K> a = fcoopmatNV<16, SCOPE, M, K>(float(gx));
    fcoopmatNV<16, SCOPE, K, N> b = fcoopmatNV<16, SCOPE, K, N>(float(lx));

    fcoopmatNV<32, SCOPE, M, N> c0 = fcoopmatNV<32, SCOPE, M, N>(float(gx));
    fcoopmatNV<32, SCOPE, M, N> c1 = fcoopmatNV<32, SCOPE, M, N>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAddNV(a, b, c0); c1 = coopMatMulAddNV(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx * (M * N), N, false);
#endif
}
)";

static const char glsl_int8_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#if ncnn_VK_KHR_cooperative_matrix
#extension GL_KHR_cooperative_matrix: require
#elif ncnn_VK_NV_cooperative_matrix
#extension GL_NV_integer_cooperative_matrix : require
#endif

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

#if ncnn_VK_KHR_cooperative_matrix
    coopmat<int8_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<int8_t, SCOPE, M, K, gl_MatrixUseA>(int8_t(gx));
    coopmat<int8_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<int8_t, SCOPE, K, N, gl_MatrixUseB>(int8_t(lx));

    coopmat<int, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<int, SCOPE, M, N, gl_MatrixUseAccumulator>(int(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
#elif ncnn_VK_NV_cooperative_matrix
    icoopmatNV<8, SCOPE, M, K> a = icoopmatNV<8, SCOPE, M, K>(int8_t(gx));
    icoopmatNV<8, SCOPE, K, N> b = icoopmatNV<8, SCOPE, K, N>(int8_t(lx));

    icoopmatNV<32, SCOPE, M, N> c = icoopmatNV<32, SCOPE, M, N>(int(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAddNV(a, b, c);)
    R"(}

    coopMatStoreNV(c, c_blob_data, gx * (M * N), N, false);
#endif
}
)";

static const char glsl_int8_matrix_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#if ncnn_VK_KHR_cooperative_matrix
#extension GL_KHR_cooperative_matrix: require
#elif ncnn_VK_NV_cooperative_matrix
#extension GL_NV_integer_cooperative_matrix : require
#endif

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

#if ncnn_VK_KHR_cooperative_matrix
    coopmat<int8_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<int8_t, SCOPE, M, K, gl_MatrixUseA>(int8_t(gx));
    coopmat<int8_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<int8_t, SCOPE, K, N, gl_MatrixUseB>(int8_t(lx));

    coopmat<int, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<int, SCOPE, M, N, gl_MatrixUseAccumulator>(int(gx));
    coopmat<int, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<int, SCOPE, M, N, gl_MatrixUseAccumulator>(int(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
#elif ncnn_VK_NV_cooperative_matrix
    icoopmatNV<8, SCOPE, M, K> a = icoopmatNV<8, SCOPE, M, K>(int8_t(gx));
    icoopmatNV<8, SCOPE, K, N> b = icoopmatNV<8, SCOPE, K, N>(int8_t(lx));

    icoopmatNV<32, SCOPE, M, N> c0 = icoopmatNV<32, SCOPE, M, N>(int(gx));
    icoopmatNV<32, SCOPE, M, N> c1 = icoopmatNV<32, SCOPE, M, N>(int(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAddNV(a, b, c0); c1 = coopMatMulAddNV(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx * (M * N), N, false);
#endif
}
)";

static const char glsl_bf16_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_bfloat16: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_matrix_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_bfloat16: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c2 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(c0);
    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c3 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(c1);

    c0 = coopmat<bfloat16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(c2 + c3);
    coopMatStore(c0, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_fp32_matrix_data[] = R"(
#version 450

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_bfloat16: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_fp32_matrix_dual_data[] = R"(
#version 450

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_bfloat16: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp8_fp16_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e4m3: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp8_fp16_matrix_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e4m3: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp8_fp32_matrix_data[] = R"(
#version 450

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e4m3: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp8_fp32_matrix_dual_data[] = R"(
#version 450

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e4m3: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate4m3_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate4m3_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf8_fp16_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e5m2: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf8_fp16_matrix_dual_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e5m2: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float16_t, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf8_fp32_matrix_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e5m2: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_16(c = coopMatMulAdd(a, b, c);)
    R"(}

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf8_fp32_matrix_dual_data[] = R"(
#version 450

#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require
#extension GL_EXT_float_e5m2: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;
layout (constant_id = 4) const int SCOPE = 3;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA> a = coopmat<floate5m2_t, SCOPE, M, K, gl_MatrixUseA>(float(gx));
    coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB> b = coopmat<floate5m2_t, SCOPE, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float, SCOPE, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {)"
        REPEAT_8(c0 = coopMatMulAdd(a, b, c0); c1 = coopMatMulAdd(a, b, c1);)
    R"(}

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static double vkpeak(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type)
{
    ncnn::VulkanDevice* vkdev = get_vulkan_device();

    if (!vkdev)
    {
        return -1;
    }

    if (!vkdev->info.support_fp16_storage() && storage_type == 1)
    {
        return 0;
    }
    if (!vkdev->info.support_fp16_storage() && storage_type == 4)
    {
        return 0;
    }
    if (!vkdev->info.support_fp16_arithmetic() && arithmetic_type == 1)
    {
        return 0;
    }
    if (!vkdev->info.support_fp16_arithmetic() && arithmetic_type == 4)
    {
        return 0;
    }
    if (!vkdev->info.support_int8_arithmetic() && arithmetic_type == 6)
    {
        return 0;
    }
    if (!vkdev->info.support_cooperative_matrix() && packing_type == 256)
    {
        return 0;
    }

    // check shader fp64 feature
    bool has_shader_fp64 = vkdev->info.physicalDevicefeatures().shaderFloat64;
    if (!has_shader_fp64 && (storage_type == 2 || arithmetic_type == 2))
    {
        return 0;
    }

    // check shader int64 feature
    bool has_shader_int64 = vkdev->info.physicalDevicefeatures().shaderInt64;
    if (!has_shader_int64 && (storage_type == 5 || arithmetic_type == 5))
    {
        return 0;
    }

    // check shader int8 dotprod feature
    bool has_shader_int8_dotprod = vkdev->info.queryShaderIntegerDotProductFeatures().shaderIntegerDotProduct;
    if (!has_shader_int8_dotprod && (arithmetic_type == 6 && packing_type == 4))
    {
        return 0;
    }

    // check shader bf16 feature
    bool has_shader_bf16 = vkdev->info.queryShaderBfloat16Features().shaderBFloat16Type;
    if (!has_shader_bf16 && (arithmetic_type == 7))
    {
        return 0;
    }

    // check shader bf16 dotprod feature
    bool has_shader_bf16_dotprod = vkdev->info.queryShaderBfloat16Features().shaderBFloat16DotProduct;
    if (!has_shader_bf16_dotprod && (arithmetic_type == 7 && packing_type == 4))
    {
        return 0;
    }

    // check shader bf16 cooperative matrix feature
    bool has_shader_bf16_matrix = vkdev->info.queryShaderBfloat16Features().shaderBFloat16CooperativeMatrix;
    if (!has_shader_bf16_matrix && (arithmetic_type == 7 && packing_type == 256))
    {
        return 0;
    }

    // check shader fp8 feature
    bool has_shader_fp8 = vkdev->info.queryShaderFloat8Features().shaderFloat8;
    if (!has_shader_fp8 && (arithmetic_type == 8 || arithmetic_type == 9))
    {
        return 0;
    }

    // check shader fp8 cooperative matrix feature
    bool has_shader_fp8_matrix = vkdev->info.queryShaderFloat8Features().shaderFloat8CooperativeMatrix;
    if (!has_shader_fp8_matrix && ((arithmetic_type == 8 || arithmetic_type == 9) && packing_type == 256))
    {
        return 0;
    }

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = storage_type == 1;
    opt.use_fp16_storage = storage_type == 1 || storage_type == 4;
    opt.use_fp16_arithmetic = arithmetic_type == 1;

    ncnn::VkAllocator* allocator = vkdev->acquire_blob_allocator();

    // reuse c storage, max 512M
    int buffer_size = std::min((int)(vkdev->get_heap_budget() / 8), 512) * 1024 * 1024;
    if (vkdev->info.type() == 1)
    {
        // max 128M for integrated gpu
        buffer_size = std::min(buffer_size, 128 * 1024 * 1024);
    }

    // sanitize buffer_size for 0.9x hardware limits
    {
        const VkPhysicalDeviceProperties& p = vkdev->info.physicalDeviceProperties();
        uint32_t max_buffer_size = (uint32_t)(p.limits.maxStorageBufferRange * 0.9) / 1024 / 1024 * 1024 * 1024;
        buffer_size = (int)std::min((uint32_t)buffer_size, max_buffer_size);
    }

    buffer_size = std::min(buffer_size, count_mb * 1024 * 1024);

    ncnn::VkMat c(buffer_size, (size_t)1u, 1, allocator);

    int elemsize;
    if (storage_type == 0 || storage_type == 3)
    {
        // fp32 / int32
        elemsize = 4;
    }
    else if (storage_type == 1 || storage_type == 4)
    {
        // fp16 / int16
        elemsize = 2;
    }
    else if (storage_type == 2 || storage_type == 5)
    {
        // fp64 / int64
        elemsize = 8;
    }
    else if (storage_type == 6)
    {
        // int8
        elemsize = 1;
    }

    int local_size_x = std::min(128, std::max(1, (int)vkdev->info.subgroup_size()));
    if (packing_type == 256)
    {
        // matrix on subgroup
        local_size_x = (int)vkdev->info.subgroup_size();
    }

    int M = 1;
    int N = 1;
    int K = 1;
    // VK_SCOPE_WORKGROUP_KHR = gl_ScopeWorkgroup = 2
    // VK_SCOPE_SUBGROUP_KHR = gl_ScopeSubgroup = 3
    int SCOPE = 3;
    bool use_fp16_fp32_matrix = false;
    bool use_bf16_fp32_matrix = false;
    bool use_fp8_fp32_matrix = false;
    if (packing_type == 256)
    {
        bool mnk_found = false;

        if (arithmetic_type == 1)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixSubProperties();

                {
                    // find fp16 * fp16 => fp16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            break;
                        }
                    }
                }

                if (!mnk_found)
                {
                    // find fp16 * fp16 => fp32
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            use_fp16_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
            else // if (vkdev->info.support_VK_NV_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesNV>& properties = vkdev->info.queryCooperativeMatrixSubPropertiesNV();

                {
                    // find fp16 * fp16 => fp16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesNV& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT16_NV
                            && (cmp.scope == VK_SCOPE_SUBGROUP_NV || cmp.scope == VK_SCOPE_WORKGROUP_NV))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            break;
                        }
                    }
                }

                if (!mnk_found)
                {
                    // find fp16 * fp16 => fp32
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesNV& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT32_NV
                            && (cmp.scope == VK_SCOPE_SUBGROUP_NV || cmp.scope == VK_SCOPE_WORKGROUP_NV))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            use_fp16_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
        }

        if (arithmetic_type == 6)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixSubProperties();

                // find int8 * int8 => int32
                for (uint32_t j = 0; j < properties.size(); j++)
                {
                    const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                    if (cmp.AType == VK_COMPONENT_TYPE_SINT8_KHR && cmp.BType == VK_COMPONENT_TYPE_SINT8_KHR
                        && cmp.CType == VK_COMPONENT_TYPE_SINT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_SINT32_KHR
                        && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                    {
                        M = cmp.MSize;
                        N = cmp.NSize;
                        K = cmp.KSize;
                        SCOPE = (int)cmp.scope;
                        mnk_found = true;
                        break;
                    }
                }
            }
            else // if (vkdev->info.support_VK_NV_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesNV>& properties = vkdev->info.queryCooperativeMatrixSubPropertiesNV();

                // find int8 * int8 => int32
                for (uint32_t j = 0; j < properties.size(); j++)
                {
                    const VkCooperativeMatrixPropertiesNV& cmp = properties[j];

                    if (cmp.AType == VK_COMPONENT_TYPE_SINT8_NV && cmp.BType == VK_COMPONENT_TYPE_SINT8_NV
                        && cmp.CType == VK_COMPONENT_TYPE_SINT32_NV && cmp.DType == VK_COMPONENT_TYPE_SINT32_NV
                        && (cmp.scope == VK_SCOPE_SUBGROUP_NV || cmp.scope == VK_SCOPE_WORKGROUP_NV))
                    {
                        M = cmp.MSize;
                        N = cmp.NSize;
                        K = cmp.KSize;
                        SCOPE = (int)cmp.scope;
                        mnk_found = true;
                        break;
                    }
                }
            }
        }

        if (arithmetic_type == 7)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixSubProperties();

                {
                    // find bf16 * bf16 => bf16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
                            && cmp.CType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmp.ResultType == VK_COMPONENT_TYPE_BFLOAT16_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            break;
                        }
                    }
                }

                if (!mnk_found)
                {
                    // find bf16 * bf16 => fp32
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            use_bf16_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
        }

        if (arithmetic_type == 8)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixSubProperties();

                {
                    // find fp8 * fp8 => fp16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT && cmp.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            break;
                        }
                    }
                }

                if (!mnk_found)
                {
                    // find fp8 * fp8 => fp32
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT && cmp.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            use_fp8_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
        }

        if (arithmetic_type == 9)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixSubProperties();

                {
                    // find bf8 * bf8 => fp16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT && cmp.BType == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            break;
                        }
                    }
                }

                if (!mnk_found)
                {
                    // find bf8 * bf8 => fp32
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT && cmp.BType == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                            && (cmp.scope == VK_SCOPE_SUBGROUP_KHR || cmp.scope == VK_SCOPE_WORKGROUP_KHR))
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            SCOPE = (int)cmp.scope;
                            mnk_found = true;
                            use_fp8_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
        }

        if (!mnk_found)
        {
            // no supported component type
            return 0;
        }
    }

    int max_invocation_count = buffer_size / elemsize;
    // make max_invocation_count be multiple of local_size_x
    max_invocation_count = std::max(max_invocation_count / local_size_x, 1) * local_size_x;
    if (packing_type == 256)
    {
        if (use_fp16_fp32_matrix || use_bf16_fp32_matrix || use_fp8_fp32_matrix)
            max_invocation_count = std::max(max_invocation_count / (M * N) / 2, 1);
        else
            max_invocation_count = std::max(max_invocation_count / (M * N), 1);
    }

    double max_gflops = 0;

    // start with little works
    int invocation_count = std::max(max_invocation_count / 32, 8);

    bool rerun = true;

    // prepare storage
    while (rerun)
    {
        rerun = false;

        // setup pipeline
        ncnn::Pipeline pipeline(vkdev);
        ncnn::Pipeline pipeline_dual(vkdev);
        {
            pipeline.set_local_size_xyz(local_size_x, 1, 1);
            pipeline_dual.set_local_size_xyz(local_size_x, 1, 1);

            std::vector<ncnn::vk_specialization_type> specializations(1);
            specializations[0].i = loop;

            // glsl to spirv
            // -1 for omit the tail '\0'
            std::vector<uint32_t> spirv;
            std::vector<uint32_t> spirv_dual;
            if (arithmetic_type == 2)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_fp64_p1_data, sizeof(glsl_fp64_p1_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_fp64_p1_dual_data, sizeof(glsl_fp64_p1_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_fp64_p4_data, sizeof(glsl_fp64_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_fp64_p4_dual_data, sizeof(glsl_fp64_p4_dual_data) - 1, opt, spirv_dual);
                }
            }
            else if (arithmetic_type == 3)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int32_p1_data, sizeof(glsl_int32_p1_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int32_p1_dual_data, sizeof(glsl_int32_p1_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int32_p4_data, sizeof(glsl_int32_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int32_p4_dual_data, sizeof(glsl_int32_p4_dual_data) - 1, opt, spirv_dual);
                }
            }
            else if (arithmetic_type == 4)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int16_p1_data, sizeof(glsl_int16_p1_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int16_p1_dual_data, sizeof(glsl_int16_p1_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int16_p4_data, sizeof(glsl_int16_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int16_p4_dual_data, sizeof(glsl_int16_p4_dual_data) - 1, opt, spirv_dual);
                }
            }
            else if (arithmetic_type == 5)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int64_p1_data, sizeof(glsl_int64_p1_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int64_p1_dual_data, sizeof(glsl_int64_p1_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int64_p4_data, sizeof(glsl_int64_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int64_p4_dual_data, sizeof(glsl_int64_p4_dual_data) - 1, opt, spirv_dual);
                }
            }
            else if (arithmetic_type == 6)
            {
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int8_p4_data, sizeof(glsl_int8_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int8_p4_dual_data, sizeof(glsl_int8_p4_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 256)
                {
                    // loop M N K SCOPE
                    specializations.resize(5);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;
                    specializations[4].i = SCOPE;

                    ncnn::compile_spirv_module(glsl_int8_matrix_data, sizeof(glsl_int8_matrix_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int8_matrix_dual_data, sizeof(glsl_int8_matrix_dual_data) - 1, opt, spirv_dual);
                }
            }
            else if (arithmetic_type == 7)
            {
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_bf16_p4_data, sizeof(glsl_bf16_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_bf16_p4_dual_data, sizeof(glsl_bf16_p4_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 256)
                {
                    // loop M N K SCOPE
                    specializations.resize(5);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;
                    specializations[4].i = SCOPE;

                    if (use_bf16_fp32_matrix)
                    {
                        ncnn::compile_spirv_module(glsl_bf16_fp32_matrix_data, sizeof(glsl_bf16_fp32_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_bf16_fp32_matrix_dual_data, sizeof(glsl_bf16_fp32_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_bf16_matrix_data, sizeof(glsl_bf16_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_bf16_matrix_dual_data, sizeof(glsl_bf16_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                }
            }
            else if (arithmetic_type == 8)
            {
                if (packing_type == 256)
                {
                    // loop M N K SCOPE
                    specializations.resize(5);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;
                    specializations[4].i = SCOPE;

                    if (use_fp8_fp32_matrix)
                    {
                        ncnn::compile_spirv_module(glsl_fp8_fp32_matrix_data, sizeof(glsl_fp8_fp32_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_fp8_fp32_matrix_dual_data, sizeof(glsl_fp8_fp32_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_fp8_fp16_matrix_data, sizeof(glsl_fp8_fp16_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_fp8_fp16_matrix_dual_data, sizeof(glsl_fp8_fp16_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                }
            }
            else if (arithmetic_type == 9)
            {
                if (packing_type == 256)
                {
                    // loop M N K SCOPE
                    specializations.resize(5);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;
                    specializations[4].i = SCOPE;

                    if (use_fp8_fp32_matrix)
                    {
                        ncnn::compile_spirv_module(glsl_bf8_fp32_matrix_data, sizeof(glsl_bf8_fp32_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_bf8_fp32_matrix_dual_data, sizeof(glsl_bf8_fp32_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_bf8_fp16_matrix_data, sizeof(glsl_bf8_fp16_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_bf8_fp16_matrix_dual_data, sizeof(glsl_bf8_fp16_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                }
            }
            else // if (arithmetic_type == 0 || arithmetic_type == 1)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_p1_data, sizeof(glsl_p1_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_p1_dual_data, sizeof(glsl_p1_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_p4_data, sizeof(glsl_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_p4_dual_data, sizeof(glsl_p4_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 256)
                {
                    // loop M N K SCOPE
                    specializations.resize(5);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;
                    specializations[4].i = SCOPE;

                    if (use_fp16_fp32_matrix)
                    {
                        ncnn::compile_spirv_module(glsl_fp16_fp32_matrix_data, sizeof(glsl_fp16_fp32_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_fp16_fp32_matrix_dual_data, sizeof(glsl_fp16_fp32_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_fp16_matrix_data, sizeof(glsl_fp16_matrix_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_fp16_matrix_dual_data, sizeof(glsl_fp16_matrix_dual_data) - 1, opt, spirv_dual);
                    }
                }
            }

            int ret0 = pipeline.create(spirv.data(), spirv.size() * 4, specializations);
            int ret1 = pipeline_dual.create(spirv_dual.data(), spirv_dual.size() * 4, specializations);
            if (ret0 != 0 || ret1 != 0)
            {
                vkdev->reclaim_blob_allocator(allocator);
                return 0;
            }
        }

        for (int i = 0; i < cmd_loop; i++)
        {
            // encode command
            ncnn::VkCompute cmd(vkdev);
            ncnn::VkCompute cmd_dual(vkdev);
            {
                std::vector<ncnn::VkMat> bindings(1);
                bindings[0] = c;

                std::vector<ncnn::vk_constant_type> constants(0);

                ncnn::VkMat dispatcher;
                dispatcher.w = invocation_count;
                dispatcher.h = 1;
                dispatcher.c = 1;
                cmd.record_pipeline(&pipeline, bindings, constants, dispatcher);
                cmd_dual.record_pipeline(&pipeline_dual, bindings, constants, dispatcher);
            }

            // time this
            {
                double t0 = ncnn::get_current_time();

                int ret = cmd.submit_and_wait();
                if (ret != 0)
                {
                    vkdev->reclaim_blob_allocator(allocator);
                    return 0;
                }

                double t1 = ncnn::get_current_time();

                double time = t1 - t0;

                if (time < 300)
                {
                    // for fast device
                    if (invocation_count * 2 <= max_invocation_count)
                    {
                        invocation_count = std::min(invocation_count * 2, max_invocation_count);
                    }
                    else
                    {
                        loop *= 2;
                    }
                    rerun = true;
                    break;
                }

                t0 = ncnn::get_current_time();

                ret = cmd_dual.submit_and_wait();
                if (ret != 0)
                {
                    vkdev->reclaim_blob_allocator(allocator);
                    return 0;
                }

                t1 = ncnn::get_current_time();

                double time_dual = t1 - t0;

                if (time_dual < 300)
                {
                    // for fast device
                    if (invocation_count * 2 <= max_invocation_count)
                    {
                        invocation_count = std::min(invocation_count * 2, max_invocation_count);
                    }
                    else
                    {
                        loop *= 2;
                    }
                    rerun = true;
                    break;
                }

                double gflops;
                {
                    double mac = (double)invocation_count * ((double)loop * 16 * 2);

                    if (packing_type == 256)
                    {
                        mac *= M * N * K;
                        mac /= local_size_x;
                    }
                    else
                    {
                        mac *= packing_type;
                    }

                    gflops = mac / time / 1000000;
                }
                double gflops_dual;
                {
                    // dual issue is faster
                    double mac = (double)invocation_count * ((double)loop * 16 * 2 + 1); // +1 for the tail c0+c1

                    if (packing_type == 256)
                    {
                        mac *= M * N * K;
                        mac /= local_size_x;
                    }
                    else
                    {
                        mac *= packing_type;
                    }

                    gflops_dual = mac / time_dual / 1000000;
                }

                gflops = std::max(gflops, gflops_dual);

                // fprintf(stderr, "%f gflops\n", gflops);

                if (gflops > max_gflops)
                    max_gflops = gflops;
            }
        }
    }

    vkdev->reclaim_blob_allocator(allocator);

    return max_gflops;
}

static double vkpeak_copy(int count_mb, int cmd_loop, int from_type, int to_type)
{
    ncnn::VulkanDevice* vkdev = get_vulkan_device();

    if (!vkdev)
    {
        return -1;
    }

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;

    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    ncnn::VkAllocator* allocator = vkdev->acquire_blob_allocator();

    opt.blob_vkallocator = allocator;
    opt.workspace_vkallocator = allocator;
    opt.staging_vkallocator = staging_allocator;

    bool d2d = from_type == 1 && to_type == 1;

    // devbuf max 512M for host and 2G for d2d
    size_t buffer_size = std::min((size_t)vkdev->get_heap_budget() / 8, d2d ? (size_t)2048 : (size_t)512) * 1024 * 1024;
    if (vkdev->info.type() == 1)
    {
        // max 128M for integrated gpu
        buffer_size = std::min(buffer_size, (size_t)128 * 1024 * 1024);
    }

    // sanitize buffer_size for 0.9x hardware limits
    {
        const VkPhysicalDeviceProperties& p = vkdev->info.physicalDeviceProperties();
        uint32_t max_buffer_size = (uint32_t)(p.limits.maxStorageBufferRange * 0.9) / 1024 / 1024 * 1024 * 1024;
        buffer_size = std::min(buffer_size, (size_t)max_buffer_size);
    }

    buffer_size = std::min(buffer_size, (size_t)count_mb * 1024 * 1024);

    double max_gbps = 0;

    if (from_type == 0 && to_type == 0)
    {
        ncnn::Mat a(1, buffer_size, 1);
        ncnn::Mat b(1, buffer_size, 1);

        for (int i = 0; i < cmd_loop; i++)
        {
            // reset cache
            memset(a, 0, buffer_size);

            ncnn::sleep(100);

            // time this
            double t0 = ncnn::get_current_time();

            memcpy(b, a, buffer_size);

            double t1 = ncnn::get_current_time();

            double time = t1 - t0;

            double gbps = buffer_size / time / 1000000;

            // fprintf(stderr, "%f gbps\n", gbps);

            if (gbps > max_gbps)
                max_gbps = gbps;
        }
    }
    if (from_type == 0 && to_type == 1)
    {
        ncnn::VkMat devbuf(1, buffer_size, 1, staging_allocator);
        ncnn::Mat hostbuf(1, buffer_size, 1);

        void* devptr = devbuf.mapped_ptr();
        void* hostptr = hostbuf.data;

        for (int i = 0; i < cmd_loop; i++)
        {
            // reset cache
            memset(hostptr, 0, buffer_size);
            staging_allocator->invalidate(devbuf.data);

            ncnn::sleep(100);

            // time this
            double t0 = ncnn::get_current_time();

            memcpy(devptr, hostptr, buffer_size);

            staging_allocator->flush(devbuf.data);

            double t1 = ncnn::get_current_time();

            double time = t1 - t0;

            double gbps = buffer_size / time / 1000000;

            // fprintf(stderr, "%f gbps\n", gbps);

            if (gbps > max_gbps)
                max_gbps = gbps;
        }
    }
    if (from_type == 1 && to_type == 0)
    {
        ncnn::VkMat devbuf(1, buffer_size, 1, staging_allocator);
        ncnn::Mat hostbuf(1, buffer_size, 1);

        void* devptr = devbuf.mapped_ptr();
        void* hostptr = hostbuf.data;

        for (int i = 0; i < cmd_loop; i++)
        {
            // reset cache
            staging_allocator->flush(devbuf.data);
            memset(hostptr, 0, buffer_size);

            ncnn::sleep(100);

            // time this
            double t0 = ncnn::get_current_time();

            staging_allocator->invalidate(devbuf.data);

            memcpy(hostptr, devptr, buffer_size);

            double t1 = ncnn::get_current_time();

            double time = t1 - t0;

            double gbps = buffer_size / time / 1000000;

            // fprintf(stderr, "%f gbps\n", gbps);

            if (gbps > max_gbps)
                max_gbps = gbps;
        }
    }
    if (from_type == 1 && to_type == 1)
    {
        ncnn::VkMat a(1, buffer_size, 1, allocator);
        ncnn::VkMat b(1, buffer_size, 1, allocator);

        for (int i = 0; i < cmd_loop * 5; i++)
        {
            // encode command
            ncnn::VkCompute cmd(vkdev);

            cmd.record_clone(a, b, opt);

            // time this
            double t0 = ncnn::get_current_time();

            int ret = cmd.submit_and_wait();
            if (ret != 0)
            {
                vkdev->reclaim_staging_allocator(staging_allocator);
                vkdev->reclaim_blob_allocator(allocator);
                return 0;
            }

            double t1 = ncnn::get_current_time();

            double time = t1 - t0;

            double gbps = buffer_size / time / 1000000;

            // fprintf(stderr, "%f gbps\n", gbps);

            if (gbps > max_gbps)
                max_gbps = gbps;
        }
    }

    vkdev->reclaim_staging_allocator(staging_allocator);
    vkdev->reclaim_blob_allocator(allocator);

    return max_gbps;
}

static bool is_adreno_6xx_7xx_8xx_series()
{
    ncnn::VulkanDevice* vkdev = get_vulkan_device();

    if (vkdev && vkdev->info.vendor_id() == 0x5143)
    {
        // qcom
        const char* device_name = vkdev->info.device_name();
        if (!device_name)
            return false;

        // momoten prefixes the underlying OpenCL device name.
        const char* adreno_name = strstr(device_name, "Adreno");
        if (!adreno_name)
            return false;

        int model = 0;
        if (sscanf(adreno_name, "Adreno (TM) %d", &model) == 1)
        {
            if (model >= 600 && model < 700)
                return true;  // Adreno 6xx
            if (model >= 700 && model < 800)
                return true;  // Adreno 7xx
            if (model >= 800 && model < 900)
                return true;  // Adreno 8xx
        }
    }

    return false;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "VkPeakNcnn", "JNI_OnLoad");

    create_vulkan(0);
    // Cache the system GPU before selecting another driver changes its name.
    g_turnip_supported = is_adreno_6xx_7xx_8xx_series();

    void* momoten = dlopen("libmomoten.so", RTLD_LOCAL | RTLD_NOW);
    if (momoten)
    {
        g_momoten_available = dlsym(momoten, "vkGetInstanceProcAddr") != 0;
        dlclose(momoten);
    }

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "VkPeakNcnn", "JNI_OnUnload");

    g_vulkan_ready = false;
    ncnn::destroy_gpu_instance();
}

// public native String GetPlatform();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetPlatform(JNIEnv* env, jobject thiz)
{
    char platform[PROP_VALUE_MAX+1];
    __system_property_get("ro.board.platform", platform);

    return env->NewStringUTF(platform);
}

// public native String GetNcnnVersion();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetNcnnVersion(JNIEnv* env, jobject thiz)
{
    return env->NewStringUTF(NCNN_VERSION_STRING);
}

// public native boolean IsVulkanSupported(int type);
JNIEXPORT jboolean JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_IsVulkanSupported(JNIEnv *env, jobject thiz, jint type)
{
    if (type == 0)
    {
        return JNI_TRUE;
    }
    if (type == 1)
    {
#if __aarch64__
        // turnip only supports adreno 6xx 7xx 8xx
        return g_turnip_supported ? JNI_TRUE : JNI_FALSE;
#else
        return JNI_FALSE;
#endif
    }

    if (type == 2)
    {
        // OpenCL device availability is checked when the driver is selected.
        return g_momoten_available ? JNI_TRUE : JNI_FALSE;
    }

    return JNI_FALSE;
}

// public native int GetVulkanType();
JNIEXPORT jint JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetVulkanType(JNIEnv *env, jobject thiz)
{
    return g_vulkan_type;
}

// public native boolean ResetVulkan(int type);
JNIEXPORT jboolean JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_ResetVulkan(JNIEnv *env, jobject thiz, jint type)
{
    if (type < 0 || type > 2)
        return JNI_FALSE;

    const int previous_type = g_vulkan_type;
    g_vulkan_ready = false;
    ncnn::destroy_gpu_instance();
    if (create_vulkan(type))
        return JNI_TRUE;

    __android_log_print(ANDROID_LOG_ERROR, "VkPeakNcnn", "Cannot initialize Vulkan driver %d; restoring %d", type, previous_type);
    g_vulkan_ready = false;
    ncnn::destroy_gpu_instance();
    create_vulkan(previous_type);
    return JNI_FALSE;
}

// public native String GetVkDevice();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetVkDevice(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = get_vulkan_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    return env->NewStringUTF(vkdev->info.device_name());
}

// public native String GetApiVersion();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetApiVersion(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = get_vulkan_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    uint32_t api_version = vkdev->info.api_version();

    char tmp[128];
    sprintf(tmp, "%u.%u.%u", VK_VERSION_MAJOR(api_version), VK_VERSION_MINOR(api_version), VK_VERSION_PATCH(api_version));

    return env->NewStringUTF(tmp);
}

// public native String GetDriverVersion();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetDriverVersion(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = get_vulkan_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    uint32_t vendor_id = vkdev->info.vendor_id();
    uint32_t driver_version = vkdev->info.driver_version();

    char tmp[128];
    if (is_momoten(vkdev->info))
    {
        // momoten encodes its release date as (year - 2000).month.day.
        sprintf(tmp, "%04u%02u%02u", VK_VERSION_MAJOR(driver_version) + 2000, VK_VERSION_MINOR(driver_version), VK_VERSION_PATCH(driver_version));
    }
    else if (vendor_id == 0x10de)
    {
        // nvidia
        sprintf(tmp, "%u.%u.%u.%u", (driver_version >> 22) & 0x3ff, (driver_version >> 14) & 0x0ff, (driver_version >> 6) & 0x0ff, (driver_version) & 0x003f);
    }
    else if (vendor_id == 0x14e4)
    {
        // broadcom
        uint32_t major = driver_version / 10000;
        uint32_t minor = (driver_version % 10000) / 100;
        sprintf(tmp, "%u.%u", major, minor);
    }
    else if (vendor_id == 0x1010)
    {
        // imagination
        if (driver_version > 500000000)
        {
            sprintf(tmp, "0.0.%u", driver_version - 500000000);
        }
        else
        {
            sprintf(tmp, "%u", driver_version);
        }
    }
    else
    {
        // use vulkan version convention
        sprintf(tmp, "%u.%u.%u", VK_VERSION_MAJOR(driver_version), VK_VERSION_MINOR(driver_version), VK_VERSION_PATCH(driver_version));
    }

    return env->NewStringUTF(tmp);
}

// public native float Run(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type);
JNIEXPORT jfloat JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_Run(JNIEnv* env, jobject thiz, jint loop, jint count_mb, jint cmd_loop, jint storage_type, jint arithmetic_type, jint packing_type)
{
    double gflops = vkpeak(loop, count_mb, cmd_loop, storage_type, arithmetic_type, packing_type);

    return (jfloat)gflops;
}

// public native float Run_copy(int count_mb, int cmd_loop, int from_type, int to_type);
JNIEXPORT jfloat JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_Run_1copy(JNIEnv* env, jobject thiz, jint count_mb, jint cmd_loop, jint from_type, jint to_type)
{
    double gbps = vkpeak_copy(count_mb, cmd_loop, from_type, to_type);

    return (jfloat)gbps;
}

}
