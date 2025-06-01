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

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

public class MainActivity extends Activity
{
    private VkPeakNcnn vkpeakncnn = new VkPeakNcnn();

    private TextView textviewModel;
    private TextView textviewAndroid;
    private TextView textviewVersion;
    private TextView textviewPlatform;
    private TextView textviewDevice;
    private TextView textviewAPI;
    private TextView textviewDriver;

    private Spinner spinnerMacs;
    private Spinner spinnerCounts;
    private Spinner spinnerLoops;

    private TextView textviewFP32;
    private TextView textviewFP32v4;
    private TextView textviewFP16;
    private TextView textviewFP16v4;
    private TextView textviewFP16mm;
    private TextView textviewFP64;
    private TextView textviewFP64v4;
    private TextView textviewINT32;
    private TextView textviewINT32v4;
    private TextView textviewINT16;
    private TextView textviewINT16v4;
    private TextView textviewINT8dp;
    private TextView textviewINT8mm;
    private TextView textviewBF16dp;
    private TextView textviewBF16mm;

    private float fp32;
    private float fp32v4;
    private float fp16;
    private float fp16v4;
    private float fp16mm;
    private float fp64;
    private float fp64v4;
    private float int32;
    private float int32v4;
    private float int16;
    private float int16v4;
    private float int8dp;
    private float int8mm;
    private float bf16dp;
    private float bf16mm;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        setTitle("ncnn Vulkan peak OPS");

        textviewModel = (TextView) findViewById(R.id.textviewModel);
        textviewAndroid = (TextView) findViewById(R.id.textviewAndroid);
        textviewVersion = (TextView) findViewById(R.id.textviewVersion);
        textviewPlatform = (TextView) findViewById(R.id.textviewPlatform);
        textviewDevice = (TextView) findViewById(R.id.textviewDevice);
        textviewAPI = (TextView) findViewById(R.id.textviewAPI);
        textviewDriver = (TextView) findViewById(R.id.textviewDriver);

        textviewModel.setText("  " + Build.MODEL);
        textviewAndroid.setText("  " + Build.VERSION.RELEASE);
        textviewVersion.setText("  ncnn-" + vkpeakncnn.GetNcnnVersion());
        textviewPlatform.setText("  " + vkpeakncnn.GetPlatform());
        textviewDevice.setText("  " + vkpeakncnn.GetVkDevice());
        textviewAPI.setText("  " + vkpeakncnn.GetApiVersion());
        textviewDriver.setText("  " + vkpeakncnn.GetDriverVersion());

        spinnerMacs = (Spinner) findViewById(R.id.spinnerMacs);
        spinnerCounts = (Spinner) findViewById(R.id.spinnerCounts);
        spinnerLoops = (Spinner) findViewById(R.id.spinnerLoops);

        textviewFP32 = (TextView) findViewById(R.id.textviewFP32);
        textviewFP32v4 = (TextView) findViewById(R.id.textviewFP32v4);
        textviewFP16 = (TextView) findViewById(R.id.textviewFP16);
        textviewFP16v4 = (TextView) findViewById(R.id.textviewFP16v4);
        textviewFP16mm = (TextView) findViewById(R.id.textviewFP16mm);
        textviewFP64 = (TextView) findViewById(R.id.textviewFP64);
        textviewFP64v4 = (TextView) findViewById(R.id.textviewFP64v4);
        textviewINT32 = (TextView) findViewById(R.id.textviewINT32);
        textviewINT32v4 = (TextView) findViewById(R.id.textviewINT32v4);
        textviewINT16 = (TextView) findViewById(R.id.textviewINT16);
        textviewINT16v4 = (TextView) findViewById(R.id.textviewINT16v4);
        textviewINT8dp = (TextView) findViewById(R.id.textviewINT8dp);
        textviewINT8mm = (TextView) findViewById(R.id.textviewINT8mm);
        textviewBF16dp = (TextView) findViewById(R.id.textviewBF16dp);
        textviewBF16mm = (TextView) findViewById(R.id.textviewBF16mm);

        // apply default settings
        spinnerMacs.setSelection(1);
        spinnerCounts.setSelection(1);
        spinnerLoops.setSelection(1);

        Button buttonRun = (Button) findViewById(R.id.buttonRun);
        buttonRun.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);

                new Thread(new Runnable() {
                    public void run() {

                        int loop = Integer.parseInt(spinnerMacs.getSelectedItem().toString());
                        int count_mb = Integer.parseInt(spinnerCounts.getSelectedItem().toString());
                        int cmd_loop = Integer.parseInt(spinnerLoops.getSelectedItem().toString());

                        sleep(500);
                        fp32 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 0, 1);
                        textviewFP32.post(new Runnable() { public void run() { textviewFP32.setText(textHelper(fp32)); } });

                        sleep(500);
                        fp32v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 0, 4);
                        textviewFP32v4.post(new Runnable() { public void run() { textviewFP32v4.setText(textHelper(fp32v4)); } });

                        sleep(500);
                        fp16 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 1, 1);
                        textviewFP16.post(new Runnable() { public void run() { textviewFP16.setText(textHelper(fp16)); } });

                        sleep(500);
                        fp16v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 1, 4);
                        textviewFP16v4.post(new Runnable() { public void run() { textviewFP16v4.setText(textHelper(fp16v4)); } });

                        sleep(500);
                        fp16mm = vkpeakncnn.Run(loop, count_mb, cmd_loop, 1, 1, 256);
                        textviewFP16mm.post(new Runnable() { public void run() { textviewFP16mm.setText(textHelper(fp16mm)); } });

                        sleep(500);
                        fp64 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 2, 2, 1);
                        textviewFP64.post(new Runnable() { public void run() { textviewFP64.setText(textHelper(fp64)); } });

                        sleep(500);
                        fp64v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 2, 2, 4);
                        textviewFP64v4.post(new Runnable() { public void run() { textviewFP64v4.setText(textHelper(fp64v4)); } });

                        sleep(500);
                        int32 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 3, 1);
                        textviewINT32.post(new Runnable() { public void run() { textviewINT32.setText(textHelper(int32)); } });

                        sleep(500);
                        int32v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 3, 4);
                        textviewINT32v4.post(new Runnable() { public void run() { textviewINT32v4.setText(textHelper(int32v4)); } });

                        sleep(500);
                        int16 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 4, 1);
                        textviewINT16.post(new Runnable() { public void run() { textviewINT16.setText(textHelper(int16)); } });

                        sleep(500);
                        int16v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 4, 4);
                        textviewINT16v4.post(new Runnable() { public void run() { textviewINT16v4.setText(textHelper(int16v4)); } });

                        sleep(500);
                        int8dp = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 5, 4);
                        textviewINT8dp.post(new Runnable() { public void run() { textviewINT8dp.setText(textHelper(int8dp)); } });

                        sleep(500);
                        int8mm = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 5, 256);
                        textviewINT8mm.post(new Runnable() { public void run() { textviewINT8mm.setText(textHelper(int8mm)); } });

                        sleep(500);
                        bf16dp = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 6, 4);
                        textviewBF16dp.post(new Runnable() { public void run() { textviewBF16dp.setText(textHelper(bf16dp)); } });

                        sleep(500);
                        bf16mm = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 6, 256);
                        textviewBF16mm.post(new Runnable() { public void run() { textviewBF16mm.setText(textHelper(bf16mm)); } });

                        textviewBF16mm.post(new Runnable() { public void run() {
                            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                        } });
                    }
                }).start();
            }
        });
    }

    private void sleep(int ms)
    {
        try
        {
            Thread.sleep(ms);
        }
        catch (InterruptedException e)
        {
            e.printStackTrace();
        }
    }

    private String textHelper(float gflops)
    {
        if (gflops == -1)
            return "  error";

        if (gflops == -233)
            return "  not supported";

        return String.format("  %.2f", gflops);
    }
}
