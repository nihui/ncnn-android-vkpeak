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
    private TextView textviewFP32v8;
    private TextView textviewFP16pv4;
    private TextView textviewFP16pv8;
    private TextView textviewFP16s;
    private TextView textviewFP16sv4;
    private TextView textviewFP16sv8;
    private TextView textviewINT32;
    private TextView textviewINT32v4;
    private TextView textviewINT16;
    private TextView textviewINT16v4;

    private float fp32;
    private float fp32v4;
    private float fp32v8;
    private float fp16pv4;
    private float fp16pv8;
    private float fp16s;
    private float fp16sv4;
    private float fp16sv8;
    private float int32;
    private float int32v4;
    private float int16;
    private float int16v4;

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
        textviewFP32v8 = (TextView) findViewById(R.id.textviewFP32v8);
        textviewFP16pv4 = (TextView) findViewById(R.id.textviewFP16pv4);
        textviewFP16pv8 = (TextView) findViewById(R.id.textviewFP16pv8);
        textviewFP16s = (TextView) findViewById(R.id.textviewFP16s);
        textviewFP16sv4 = (TextView) findViewById(R.id.textviewFP16sv4);
        textviewFP16sv8 = (TextView) findViewById(R.id.textviewFP16sv8);
        textviewINT32 = (TextView) findViewById(R.id.textviewINT32);
        textviewINT32v4 = (TextView) findViewById(R.id.textviewINT32v4);
        textviewINT16 = (TextView) findViewById(R.id.textviewINT16);
        textviewINT16v4 = (TextView) findViewById(R.id.textviewINT16v4);

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
                        fp32 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 0, 0);
                        textviewFP32.post(new Runnable() { public void run() { textviewFP32.setText(textHelper(fp32)); } });

                        sleep(500);
                        fp32v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 0, 1);
                        textviewFP32v4.post(new Runnable() { public void run() { textviewFP32v4.setText(textHelper(fp32v4)); } });

                        sleep(500);
                        fp32v8 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 0, 0, 2);
                        textviewFP32v8.post(new Runnable() { public void run() { textviewFP32v8.setText(textHelper(fp32v8)); } });

                        sleep(500);
                        fp16pv4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 1, 1, 1);
                        textviewFP16pv4.post(new Runnable() { public void run() { textviewFP16pv4.setText(textHelper(fp16pv4)); } });

                        sleep(500);
                        fp16pv8 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 1, 1, 2);
                        textviewFP16pv8.post(new Runnable() { public void run() { textviewFP16pv8.setText(textHelper(fp16pv8)); } });

                        sleep(500);
                        fp16s = vkpeakncnn.Run(loop, count_mb, cmd_loop, 2, 1, 0);
                        textviewFP16s.post(new Runnable() { public void run() { textviewFP16s.setText(textHelper(fp16s)); } });

                        sleep(500);
                        fp16sv4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 2, 1, 1);
                        textviewFP16sv4.post(new Runnable() { public void run() { textviewFP16sv4.setText(textHelper(fp16sv4)); } });

                        sleep(500);
                        fp16sv8 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 2, 1, 2);
                        textviewFP16sv8.post(new Runnable() { public void run() { textviewFP16sv8.setText(textHelper(fp16sv8)); } });

                        sleep(500);
                        int32 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 2, 0);
                        textviewINT32.post(new Runnable() { public void run() { textviewINT32.setText(textHelper(int32)); } });

                        sleep(500);
                        int32v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 3, 2, 1);
                        textviewINT32v4.post(new Runnable() { public void run() { textviewINT32v4.setText(textHelper(int32v4)); } });

                        sleep(500);
                        int16 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 4, 3, 0);
                        textviewINT16.post(new Runnable() { public void run() { textviewINT16.setText(textHelper(int16)); } });

                        sleep(500);
                        int16v4 = vkpeakncnn.Run(loop, count_mb, cmd_loop, 4, 3, 1);
                        textviewINT16v4.post(new Runnable() { public void run() { textviewINT16v4.setText(textHelper(int16v4)); } });

                        textviewINT16v4.post(new Runnable() { public void run() {
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
