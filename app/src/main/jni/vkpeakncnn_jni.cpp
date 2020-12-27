// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <vulkan/vulkan.h>

// ncnn
#include "benchmark.h"
#include "c_api.h"
#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "pipeline.h"

static const char glsl_p1_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (constant_id = 0) const int count = 0;
layout (constant_id = 1) const int loop = 1;

layout (binding = 0) readonly buffer a_blob { sfp a_blob_data[]; };
layout (binding = 1) readonly buffer b_blob { sfp b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { sfp c_blob_data[]; };

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);

    if (gx >= count || gy >= 1 || gz >= 1)
        return;

    afp a = buffer_ld1(a_blob_data, gx);
    afp b = buffer_ld1(b_blob_data, gx);

    afp c = afp(1.f);

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
    }

    buffer_st1(c_blob_data, gx, c);
}
)";

static const char glsl_p4_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (constant_id = 0) const int count = 0;
layout (constant_id = 1) const int loop = 1;

layout (binding = 0) readonly buffer a_blob { sfpvec4 a_blob_data[]; };
layout (binding = 1) readonly buffer b_blob { sfpvec4 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { sfpvec4 c_blob_data[]; };

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);

    if (gx >= count || gy >= 1 || gz >= 1)
        return;

    afpvec4 a = buffer_ld4(a_blob_data, gx);
    afpvec4 b = buffer_ld4(b_blob_data, gx);

    afpvec4 c = afpvec4(1.f);

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
    }

    buffer_st4(c_blob_data, gx, c);
}
)";

static const char glsl_p8_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (constant_id = 0) const int count = 0;
layout (constant_id = 1) const int loop = 1;

layout (binding = 0) readonly buffer a_blob { sfpvec8 a_blob_data[]; };
layout (binding = 1) readonly buffer b_blob { sfpvec8 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { sfpvec8 c_blob_data[]; };

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);

    if (gx >= count || gy >= 1 || gz >= 1)
        return;

    afpvec8 a = buffer_ld8(a_blob_data, gx);
    afpvec8 b = buffer_ld8(b_blob_data, gx);

    afpvec8 c = afpvec8(afpvec4(1.f), afpvec4(1.f));

    for (int i = 0; i < loop; i++)
    {
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
        c[0] = a[0] * c[0] + b[0];
        c[1] = a[1] * c[1] + b[1];
    }

    buffer_st8(c_blob_data, gx, c);
}
)";

static float vkpeak(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type)
{
    const int count = count_mb * 1024 * 1024;

    int elempack = packing_type == 0 ? 1 : packing_type == 1 ? 4 : 8;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    if (!vkdev)
    {
        return -1;
    }

    if (!vkdev->info.support_fp16_storage && storage_type == 2)
    {
        return -233;
    }
    if (!vkdev->info.support_fp16_arithmetic && arithmetic_type == 1)
    {
        return -233;
    }

    double max_gflops = -233;

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = storage_type == 1;
    opt.use_fp16_storage = storage_type == 2;
    opt.use_fp16_arithmetic = arithmetic_type == 1;
    opt.use_shader_pack8 = packing_type == 2;

    // setup pipeline
    ncnn::Pipeline pipeline(vkdev);
    {
        int local_size_x = std::min(128, std::max(32, (int)vkdev->info.subgroup_size));

        pipeline.set_local_size_xyz(local_size_x, 1, 1);

        std::vector<ncnn::vk_specialization_type> specializations(2);
        specializations[0].i = count;
        specializations[1].i = loop;

        // glsl to spirv
        // -1 for omit the tail '\0'
        std::vector<uint32_t> spirv;
        if (packing_type == 0)
        {
            ncnn::compile_spirv_module(glsl_p1_data, sizeof(glsl_p1_data) - 1, opt, spirv);
        }
        if (packing_type == 1)
        {
            ncnn::compile_spirv_module(glsl_p4_data, sizeof(glsl_p4_data) - 1, opt, spirv);
        }
        if (packing_type == 2)
        {
            ncnn::compile_spirv_module(glsl_p8_data, sizeof(glsl_p8_data) - 1, opt, spirv);
        }

        pipeline.create(spirv.data(), spirv.size() * 4, specializations);
    }

    ncnn::VkAllocator* allocator = vkdev->acquire_blob_allocator();

    // prepare storage
    {
    ncnn::VkMat a;
    ncnn::VkMat b;
    ncnn::VkMat c;
    {
        if (opt.use_fp16_packed || opt.use_fp16_storage)
        {
            a.create(count, (size_t)(2u * elempack), elempack, allocator);
            b.create(count, (size_t)(2u * elempack), elempack, allocator);
            c.create(count, (size_t)(2u * elempack), elempack, allocator);
        }
        else
        {
            a.create(count, (size_t)(4u * elempack), elempack, allocator);
            b.create(count, (size_t)(4u * elempack), elempack, allocator);
            c.create(count, (size_t)(4u * elempack), elempack, allocator);
        }
    }

    for (int i = 0; i < cmd_loop; i++)
    {
        // encode command
        ncnn::VkCompute cmd(vkdev);
        {
            std::vector<ncnn::VkMat> bindings(3);
            bindings[0] = a;
            bindings[1] = b;
            bindings[2] = c;

            std::vector<ncnn::vk_constant_type> constants(0);

            cmd.record_pipeline(&pipeline, bindings, constants, c);
        }

        // time this
        {
            double t0 = ncnn::get_current_time();

            int ret = cmd.submit_and_wait();
            if (ret != 0)
            {
                vkdev->reclaim_blob_allocator(allocator);
                return -1;
            }

            double time = ncnn::get_current_time() - t0;

            const double mac = (double)count * (double)loop * 8 * elempack * 2;

            double gflops = mac / time / 1000000;

//             fprintf(stderr, "%f gflops\n", gflops);

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
    return env->NewStringUTF(ncnn_version());
}

// public native String GetVkDevice();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetVkDevice(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    return env->NewStringUTF(vkdev->info.device_name.c_str());
}

// public native String GetApiVersion();
JNIEXPORT jstring JNICALL Java_com_tencent_vkpeakncnn_VkPeakNcnn_GetApiVersion(JNIEnv* env, jobject thiz)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        return env->NewStringUTF("No vulkan device");
    }

    uint32_t api_version = vkdev->info.api_version;

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

    uint32_t driver_version = vkdev->info.driver_version;

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
