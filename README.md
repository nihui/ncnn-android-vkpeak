# ncnn-android-vkpeak

![download](https://img.shields.io/github/downloads/nihui/ncnn-android-vkpeak/total.svg)

The ncnn android vulkan peak FLOPS app

## download apk

prebuild apk file can be download from release page

https://github.com/nihui/ncnn-android-vkpeak/releases

## how to build and run
### step1
clone this project with submodules

### step2
https://github.com/nihui/mesa-turnip-android-driver

* Download mesa-turnip-android-XYZ.zip
* Create directory **app/src/main/jniLibs/arm64-v8a** if not exists
* Extract `libvulkan_freedreno.so` from mesa-turnip-android-XYZ.zip into **app/src/main/jniLibs/arm64-v8a**

### step3
https://github.com/nihui/momoten/releases/tag/20261020

* Download the momoten 20261020 Android runtime zip for each ABI:
  * `arm64-v8a`: `momoten-20261020-android-arm64-v8a-api21.zip`
  * `armeabi-v7a`: `momoten-20261020-android-armeabi-v7a-api14.zip`
  * `x86`: `momoten-20261020-android-x86-api14.zip`
  * `x86_64`: `momoten-20261020-android-x86_64-api21.zip`
* Extract `lib/libmomoten.so` into **app/src/main/jniLibs/ABI/libmomoten.so** (replace ABI with the corresponding ABI above, without an extra `lib` directory).
* Copy `LICENSE.txt` and `THIRD_PARTY_NOTICES.md` from one of the archives into **app/src/main/assets/licenses/momoten**.
* The release workflow downloads all four archives, verifies their SHA-256 checksums, and bundles the libraries and notices automatically.

### step4
open this project with Android Studio, build it and enjoy!

## drivers

Select **system driver**, **mesa turnip**, or **momoten** in the driver menu.
Turnip remains limited to supported arm64 Adreno GPUs. momoten translates Vulkan
compute to OpenCL and requires an accessible, compatible OpenCL device; including
the library does not guarantee support on every phone. It is experimental and is
not a conformant Vulkan implementation.

If a selected driver cannot initialize, the app reports the failure and attempts
to restore the previous driver. The selector and version reflect the actual loaded
driver, including ncnn's automatic momoten fallback. momoten's version is shown as
its release date, for example `20261020`.

The app still requires Android 7.0 (API 24) or newer and NDK 30. The API 14/21 labels
on the momoten archives describe those libraries, not the app's minimum Android
version; Android 4 support is not included.

## screenshot
![](screenshot.png)

