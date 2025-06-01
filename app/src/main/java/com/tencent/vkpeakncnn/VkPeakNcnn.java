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

import android.content.res.AssetManager;

public class VkPeakNcnn
{
    public native String GetPlatform();

    public native String GetNcnnVersion();

    public native String GetVkDevice();
    public native String GetApiVersion();
    public native String GetDriverVersion();

    // device_id        = 0
    // storage_type     = 0/1/2/3/4/5/6 = fp32 fp16 fp64 int32 int16 int8 bf16
    // arithmetic_type  = 0/1/2/3/4/5/6 = fp32 fp16 fp64 int32 int16 int8 bf16
    // packing_type     = 1/4/256       = scalar vec4/dotprod matrix
    public native float Run(int loop, int count_mb, int cmd_loop, int storage_type, int arithmetic_type, int packing_type);

    static {
        System.loadLibrary("vkpeakncnn");
    }
}
