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

package com.tencent.vkpeakncnn;

public class VkPeakNcnn
{
    public native String GetPlatform();

    public native String GetNcnnVersion();

    public native boolean IsVulkanSupported(int type);
    public native int GetVulkanType();
    public native boolean ResetVulkan(int type);

    public native String GetVkDevice();
    public native String GetApiVersion();
    public native String GetDriverVersion();

    // storage_type / arithmetic_type
    //      0 = fp32
    //      1 = fp16
    //      2 = fp64
    //      3 = int32
    //      4 = int16
    //      5 = int64
    //      6 = int8
    //      7 = bf16
    //      8 = fp8
    //      9 = bf8

    // packing_type
    //      1 = scalar
    //      4 = vec4 / dotprod
    //    256 = matrix

    public native float Run(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type);

    // device_type
    //      0 = cpu
    //      1 = gpu

    public native float Run_copy(int count_mb, int cmd_loop, int from_type, int to_type);

    static {
        System.loadLibrary("vkpeakncnn");
    }
}
