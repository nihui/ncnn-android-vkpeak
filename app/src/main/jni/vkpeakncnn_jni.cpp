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

#include <float.h>
#include <string>
#include <vector>

// ncnn
#include <benchmark.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    c_blob_data[gx] = sfp(c);
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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = sfp(c0);
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

    afpvec4 c = afpvec4(gx);

    afpvec4 a = c + afpvec4(0,1,2,-3);
    afpvec4 b = afpvec4(lx) + afpvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    c_blob_data[gx] = sfp((c[0] + c[1]) + (c[2] + c[3]));
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

    afpvec4 c0 = afpvec4(gx);
    afpvec4 c1 = afpvec4(lx);

    afpvec4 a = c0 + afpvec4(0,1,2,-3);
    afpvec4 b = c1 + afpvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = sfp((c0[0] + c0[1]) + (c0[2] + c0[3]));
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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

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
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

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
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = int((c0[0] + c0[1]) + (c0[2] + c0[3]));
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
    {
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
        c = dotPacked4x8AccSatEXT(a, b, c);
    }

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
    {
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
        c0 = dotPacked4x8AccSatEXT(a, b, c0);
        c1 = dotPacked4x8AccSatEXT(a, b, c1);
    }

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
    {
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.x = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.y = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.z = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.w = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.x = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.y = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.z = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.w = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.x = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.y = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.z = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.w = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.x = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.y = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.z = bfloat16BitsToUintEXT(c);
        c = dot(uintBitsToBFloat16EXT(a), b);
        a.w = bfloat16BitsToUintEXT(c);
    }

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
    {
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.x = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.y = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.z = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.w = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.x = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.y = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.z = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.w = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.x = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.y = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.z = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.w = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.x = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.y = bfloat16BitsToUintEXT(c1);
        c0 = dot(uintBitsToBFloat16EXT(a0), b);
        a0.z = bfloat16BitsToUintEXT(c0);
        c1 = dot(uintBitsToBFloat16EXT(a1), b);
        a1.w = bfloat16BitsToUintEXT(c1);
    }

    c_blob_data[gx] = float(c0) + float(c1);
}
)";

static const char glsl_fp16_matrix_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    fcoopmatNV<16, gl_ScopeSubgroup, M, K> a = fcoopmatNV<16, gl_ScopeSubgroup, M, K>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, K, N> b = fcoopmatNV<16, gl_ScopeSubgroup, K, N>(float(lx));

    fcoopmatNV<16, gl_ScopeSubgroup, M, N> c = fcoopmatNV<16, gl_ScopeSubgroup, M, N>(float(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
    }

    coopMatStoreNV(c, c_blob_data, gx * (M * N) / 2, N / 2, false);
}
)";

static const char glsl_fp16_matrix_dual_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    fcoopmatNV<16, gl_ScopeSubgroup, M, K> a = fcoopmatNV<16, gl_ScopeSubgroup, M, K>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, K, N> b = fcoopmatNV<16, gl_ScopeSubgroup, K, N>(float(lx));

    fcoopmatNV<16, gl_ScopeSubgroup, M, N> c0 = fcoopmatNV<16, gl_ScopeSubgroup, M, N>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, M, N> c1 = fcoopmatNV<16, gl_ScopeSubgroup, M, N>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx * (M * N) / 2, N / 2, false);
}
)";

static const char glsl_fp16_matrix_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c = coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
    }

    coopMatStore(c, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp16_matrix_dual_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp16_fp32_matrix_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    fcoopmatNV<16, gl_ScopeSubgroup, M, K> a = fcoopmatNV<16, gl_ScopeSubgroup, M, K>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, K, N> b = fcoopmatNV<16, gl_ScopeSubgroup, K, N>(float(lx));

    fcoopmatNV<32, gl_ScopeSubgroup, M, N> c = fcoopmatNV<32, gl_ScopeSubgroup, M, N>(float(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
    }

    coopMatStoreNV(c, c_blob_data, gx * (M * N), N, false);
}
)";

static const char glsl_fp16_fp32_matrix_dual_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    fcoopmatNV<16, gl_ScopeSubgroup, M, K> a = fcoopmatNV<16, gl_ScopeSubgroup, M, K>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, K, N> b = fcoopmatNV<16, gl_ScopeSubgroup, K, N>(float(lx));

    fcoopmatNV<32, gl_ScopeSubgroup, M, N> c0 = fcoopmatNV<32, gl_ScopeSubgroup, M, N>(float(gx));
    fcoopmatNV<32, gl_ScopeSubgroup, M, N> c1 = fcoopmatNV<32, gl_ScopeSubgroup, M, N>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx * (M * N), N, false);
}
)";

static const char glsl_fp16_fp32_matrix_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
    }

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_fp16_fp32_matrix_dual_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_int8_matrix_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require
#extension GL_NV_integer_cooperative_matrix : require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    icoopmatNV<8, gl_ScopeSubgroup, M, K> a = icoopmatNV<8, gl_ScopeSubgroup, M, K>(int8_t(gx));
    icoopmatNV<8, gl_ScopeSubgroup, K, N> b = icoopmatNV<8, gl_ScopeSubgroup, K, N>(int8_t(lx));

    icoopmatNV<32, gl_ScopeSubgroup, M, N> c = icoopmatNV<32, gl_ScopeSubgroup, M, N>(int(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
        c = coopMatMulAddNV(a, b, c);
    }

    coopMatStoreNV(c, c_blob_data, gx * (M * N), N, false);
}
)";

static const char glsl_int8_matrix_dual_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require
#extension GL_NV_integer_cooperative_matrix : require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    icoopmatNV<8, gl_ScopeSubgroup, M, K> a = icoopmatNV<8, gl_ScopeSubgroup, M, K>(int8_t(gx));
    icoopmatNV<8, gl_ScopeSubgroup, K, N> b = icoopmatNV<8, gl_ScopeSubgroup, K, N>(int8_t(lx));

    icoopmatNV<32, gl_ScopeSubgroup, M, N> c0 = icoopmatNV<32, gl_ScopeSubgroup, M, N>(int(gx));
    icoopmatNV<32, gl_ScopeSubgroup, M, N> c1 = icoopmatNV<32, gl_ScopeSubgroup, M, N>(int(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
        c0 = coopMatMulAddNV(a, b, c0);
        c1 = coopMatMulAddNV(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx * (M * N), N, false);
}
)";

static const char glsl_int8_matrix_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<int8_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<int8_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(int8_t(gx));
    coopmat<int8_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<int8_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(int8_t(lx));

    coopmat<int, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c = coopmat<int, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(int(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
    }

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_int8_matrix_dual_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<int8_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<int8_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(int8_t(gx));
    coopmat<int8_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<int8_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(int8_t(lx));

    coopmat<int, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c0 = coopmat<int, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(int(gx));
    coopmat<int, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c1 = coopmat<int, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(int(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_matrix_khr_data[] = R"(
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

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c = coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
    }

    coopMatStore(c, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_matrix_dual_khr_data[] = R"(
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

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c0 = coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c1 = coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
    }

    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c2 = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(c0);
    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c3 = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(c1);

    c0 = coopmat<bfloat16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(c2 + c3);
    coopMatStore(c0, c_blob_data, gx * (M * N) / 2, N / 2, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_fp32_matrix_khr_data[] = R"(
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

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));

    for (int i = 0; i < loop; i++)
    {
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
        c = coopMatMulAdd(a, b, c);
    }

    coopMatStore(c, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static const char glsl_bf16_fp32_matrix_dual_khr_data[] = R"(
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

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<bfloat16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<bfloat16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx * (M * N), N, gl_CooperativeMatrixLayoutRowMajor);
}
)";

static double vkpeak(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    if (!vkdev)
    {
        return 0;
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
    if (!vkdev->info.support_int8_arithmetic() && arithmetic_type == 5)
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

    // check shader int8 dotprod feature
    bool has_shader_int8_dotprod = vkdev->info.queryShaderIntegerDotProductFeatures().shaderIntegerDotProduct;
    if (!has_shader_int8_dotprod && (arithmetic_type == 5 && packing_type == 4))
    {
        return 0;
    }

    // check shader bf16 feature
    bool has_shader_bf16 = vkdev->info.queryShaderBfloat16Features().shaderBFloat16Type;
    if (!has_shader_bf16 && (arithmetic_type == 6))
    {
        return 0;
    }

    // check shader bf16 dotprod feature
    bool has_shader_bf16_dotprod = vkdev->info.queryShaderBfloat16Features().shaderBFloat16DotProduct;
    if (!has_shader_bf16_dotprod && (arithmetic_type == 6 && packing_type == 4))
    {
        return 0;
    }

    // check shader bf16 cooperative matrix feature
    bool has_shader_bf16_matrix = vkdev->info.queryShaderBfloat16Features().shaderBFloat16CooperativeMatrix;
    if (!has_shader_bf16_matrix && (arithmetic_type == 6 && packing_type == 256))
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
    else if (storage_type == 2)
    {
        // fp64
        elemsize = 8;
    }
    else if (storage_type == 5)
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
    bool use_fp16_fp32_matrix = false;
    bool use_bf16_fp32_matrix = false;
    if (packing_type == 256)
    {
        bool mnk_found = false;

        if (arithmetic_type == 1)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixProperties();

                {
                    // find fp16 * fp16 => fp16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR
                            && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
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
                            && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            mnk_found = true;
                            use_fp16_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
            else // if (vkdev->info.support_VK_NV_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesNV>& properties = vkdev->info.queryCooperativeMatrixPropertiesNV();

                {
                    // find fp16 * fp16 => fp16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesNV& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                            && cmp.CType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT16_NV
                            && cmp.scope == VK_SCOPE_SUBGROUP_NV)
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
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
                            && cmp.scope == VK_SCOPE_SUBGROUP_NV)
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            mnk_found = true;
                            use_fp16_fp32_matrix = true;
                            break;
                        }
                    }
                }
            }
        }

        if (arithmetic_type == 5)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixProperties();

                // find int8 * int8 => int32
                for (uint32_t j = 0; j < properties.size(); j++)
                {
                    const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                    if (cmp.AType == VK_COMPONENT_TYPE_SINT8_KHR && cmp.BType == VK_COMPONENT_TYPE_SINT8_KHR
                        && cmp.CType == VK_COMPONENT_TYPE_SINT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_SINT32_KHR
                        && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
                    {
                        M = cmp.MSize;
                        N = cmp.NSize;
                        K = cmp.KSize;
                        mnk_found = true;
                        break;
                    }
                }
            }
            else // if (vkdev->info.support_VK_NV_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesNV>& properties = vkdev->info.queryCooperativeMatrixPropertiesNV();

                // find int8 * int8 => int32
                for (uint32_t j = 0; j < properties.size(); j++)
                {
                    const VkCooperativeMatrixPropertiesNV& cmp = properties[j];

                    if (cmp.AType == VK_COMPONENT_TYPE_SINT8_NV && cmp.BType == VK_COMPONENT_TYPE_SINT8_NV
                        && cmp.CType == VK_COMPONENT_TYPE_SINT32_NV && cmp.DType == VK_COMPONENT_TYPE_SINT32_NV
                        && cmp.scope == VK_SCOPE_SUBGROUP_NV)
                    {
                        M = cmp.MSize;
                        N = cmp.NSize;
                        K = cmp.KSize;
                        mnk_found = true;
                        break;
                    }
                }
            }
        }

        if (arithmetic_type == 6)
        {
            if (vkdev->info.support_VK_KHR_cooperative_matrix())
            {
                const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixProperties();

                {
                    // find bf16 * bf16 => bf16
                    for (uint32_t j = 0; j < properties.size(); j++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                        if (cmp.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
                            && cmp.CType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmp.ResultType == VK_COMPONENT_TYPE_BFLOAT16_KHR
                            && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
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
                            && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
                        {
                            M = cmp.MSize;
                            N = cmp.NSize;
                            K = cmp.KSize;
                            mnk_found = true;
                            use_bf16_fp32_matrix = true;
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
        if (use_fp16_fp32_matrix || use_bf16_fp32_matrix)
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
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int8_p4_data, sizeof(glsl_int8_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_int8_p4_dual_data, sizeof(glsl_int8_p4_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 256)
                {
                    // loop M N K
                    specializations.resize(4);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;

                    if (vkdev->info.support_VK_KHR_cooperative_matrix())
                    {
                        ncnn::compile_spirv_module(glsl_int8_matrix_khr_data, sizeof(glsl_int8_matrix_khr_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_int8_matrix_dual_khr_data, sizeof(glsl_int8_matrix_dual_khr_data) - 1, opt, spirv_dual);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_int8_matrix_nv_data, sizeof(glsl_int8_matrix_nv_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_int8_matrix_dual_nv_data, sizeof(glsl_int8_matrix_dual_nv_data) - 1, opt, spirv_dual);
                    }
                }
            }
            else if (arithmetic_type == 6)
            {
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_bf16_p4_data, sizeof(glsl_bf16_p4_data) - 1, opt, spirv);
                    ncnn::compile_spirv_module(glsl_bf16_p4_dual_data, sizeof(glsl_bf16_p4_dual_data) - 1, opt, spirv_dual);
                }
                if (packing_type == 256)
                {
                    // loop M N K
                    specializations.resize(4);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;

                    if (use_bf16_fp32_matrix)
                    {
                        ncnn::compile_spirv_module(glsl_bf16_fp32_matrix_khr_data, sizeof(glsl_bf16_fp32_matrix_khr_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_bf16_fp32_matrix_dual_khr_data, sizeof(glsl_bf16_fp32_matrix_dual_khr_data) - 1, opt, spirv_dual);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_bf16_matrix_khr_data, sizeof(glsl_bf16_matrix_khr_data) - 1, opt, spirv);
                        ncnn::compile_spirv_module(glsl_bf16_matrix_dual_khr_data, sizeof(glsl_bf16_matrix_dual_khr_data) - 1, opt, spirv_dual);
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
                    // loop M N K
                    specializations.resize(4);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;

                    if (vkdev->info.support_VK_KHR_cooperative_matrix())
                    {
                        if (use_fp16_fp32_matrix)
                        {
                            ncnn::compile_spirv_module(glsl_fp16_fp32_matrix_khr_data, sizeof(glsl_fp16_fp32_matrix_khr_data) - 1, opt, spirv);
                            ncnn::compile_spirv_module(glsl_fp16_fp32_matrix_dual_khr_data, sizeof(glsl_fp16_fp32_matrix_dual_khr_data) - 1, opt, spirv_dual);
                        }
                        else
                        {
                            ncnn::compile_spirv_module(glsl_fp16_matrix_khr_data, sizeof(glsl_fp16_matrix_khr_data) - 1, opt, spirv);
                            ncnn::compile_spirv_module(glsl_fp16_matrix_dual_khr_data, sizeof(glsl_fp16_matrix_dual_khr_data) - 1, opt, spirv_dual);
                        }
                    }
                    else
                    {
                        if (use_fp16_fp32_matrix)
                        {
                            ncnn::compile_spirv_module(glsl_fp16_fp32_matrix_nv_data, sizeof(glsl_fp16_fp32_matrix_nv_data) - 1, opt, spirv);
                            ncnn::compile_spirv_module(glsl_fp16_fp32_matrix_dual_nv_data, sizeof(glsl_fp16_fp32_matrix_dual_nv_data) - 1, opt, spirv_dual);
                        }
                        else
                        {
                            ncnn::compile_spirv_module(glsl_fp16_matrix_nv_data, sizeof(glsl_fp16_matrix_nv_data) - 1, opt, spirv);
                            ncnn::compile_spirv_module(glsl_fp16_matrix_dual_nv_data, sizeof(glsl_fp16_matrix_dual_nv_data) - 1, opt, spirv_dual);
                        }
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

                ret = cmd_dual.submit_and_wait();
                if (ret != 0)
                {
                    vkdev->reclaim_blob_allocator(allocator);
                    return 0;
                }

                double t2 = ncnn::get_current_time();

                double time = t1 - t0;
                double time_dual = t2 - t1;

                if (time < 800 || time_dual < 800)
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

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "VkPeakNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "VkPeakNcnn", "JNI_OnUnload");

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

// public native String GetVkDevice();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetVkDevice(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    return env->NewStringUTF(vkdev->info.device_name());
}

// public native String GetApiVersion();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetApiVersion(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
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
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    uint32_t driver_version = vkdev->info.driver_version();

    char tmp[128];
    sprintf(tmp, "%u.%u.%u", VK_VERSION_MAJOR(driver_version), VK_VERSION_MINOR(driver_version), VK_VERSION_PATCH(driver_version));

    return env->NewStringUTF(tmp);
}

// public native float Run(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type);
JNIEXPORT jfloat JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_Run(JNIEnv* env, jobject thiz, jint loop, jint count_mb, jint cmd_loop, jint storage_type, jint arithmetic_type, jint packing_type)
{
    double gflops = vkpeak(loop, count_mb, cmd_loop, storage_type, arithmetic_type, packing_type);

    return (jfloat)gflops;
}

}
