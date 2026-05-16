# 拉流

> 拉流指的是从某个网络源（通常是一个实时流媒体服务器）获取媒体流（视频或音频流）并进行处理或保存的过程。


### 方案一：使用opencv直接处理摄像头视频流

opencv本身就可以直接与摄像头交互，而其实Opencv内部调用了FFmpeg库来处理视频流。

```python
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 显示图像
    cv2.imshow('frame', frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
```

### 方案二： 利用libuvc进行拉流

### 1. 安装库文件 libuvc.so
```bash
git clone https://github.com/libuvc/libuvc.git
cd libuvc
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```

!!! tip
    上面编译过程中需要安装libusb-1.0
    `sudo apt-get install libusb-1.0-0-dev`
    如果存在版本依赖问题，强制安装与libusb-1匹配的版本环境。`sudo apt-get install libusb-1.0-0=2:1.0.25-1ubuntu1`


### 2.设置USB权限
>当插入一个新的摄像头设备时，系统将自动按照这条规则调整设备的用户组和权限，使得所有用户都可以访问设备，而不再需要使用 sudo。这样，可以在不使用 sudo 的情况下运行程序来访问摄像头。

* 在`/etc/udev/rules.d`下新建文件`99-usbcam.rules`
* 打开文件`sudo vim 99-usbcam.rules`, 添加`SUBSYSTEMS=="usb", ENV{DEVTYPE}=="usb_device", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="08cc", MODE="0666"`
* 保存文件并退出
* 重新加载udev规则：
  * `sudo udevadm control --reload-rules`
  * `sudo udevadm trigger`
这将重新加载所有的`udev`规则，并立即应用新的规则.


