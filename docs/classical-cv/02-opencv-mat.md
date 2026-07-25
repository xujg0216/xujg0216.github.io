# cv::Mat 容器


`cv::Mat` 是 OpenCV 中用来存储图像、图像区域、或者任意矩阵（如特征矩阵、变换矩阵等）的类。

* 全称：**Matrix（矩阵）**
* 本质上是：一个包含数据（Data）、尺寸（Size）、类型（Type）等信息的二维数据结构。
* 类定义在：`opencv2/core/mat.hpp`


```cpp
class Mat {
public:
    uchar* data;         // 实际的数据指针
    int rows, cols;      // 行数、列数（高、宽）
    int type();          // 数据类型（如 CV_8UC3）
    ...
};
```

> `cv::Mat` 使用 **智能引用计数**。多个 `Mat` 共享一份数据，直到有人 `.clone()`。


## 1. 常见创建方式

**空矩阵**

```cpp
cv::Mat m1;
```

**指定大小和类型**

```cpp
cv::Mat m2(480, 640, CV_8UC1);        // 单通道灰度图像
cv::Mat m3(480, 640, CV_8UC3, cv::Scalar(0, 0, 255));  // 红色背景
```

**从已有数据构造**

```cpp
uchar data[] = {1, 2, 3, 4, 5, 6};
cv::Mat m4(2, 3, CV_8UC1, data);  // 2行3列
```

**采用matlab方式初始化**

```cpp
cv::Mat E = cv::Mat::eye(4, 4, CV_64F);
cv::Mat O = cv::Mat::ones(2, 2, CV_32F);
cv::Mat z = cv::Mat::zeros(3, 3, CV_8UC1);  
```

## 2. 读取图像

```cpp
cv::Mat img = cv::imread("image.jpg");  // 自动检测通道
```

## 3. 复制与深拷贝

```cpp
cv::Mat copy1 = img;       // 浅拷贝，数据共享，增加数据的引用次数，指向同一块地址
cv::Mat copy2 = img.clone(); // 深拷贝
```


## 4. 常用方法

### 4.1 基本属性

```cpp
img.rows;       // 高度
img.cols;       // 宽度
img.channels(); // 通道数
img.type();     // 数据类型（CV_8UC3等）
img.depth();    // 深度（CV_8U、CV_32F 等）
```

### 4.2 判断图像有效性

```cpp
if (img.empty()) {
    std::cerr << "图像为空！\n";
}
```

### 4.3 访问像素

```cpp
// 灰度图像
uchar gray = img.at<uchar>(y, x);

// 彩色图像
cv::Vec3b color = img.at<cv::Vec3b>(y, x);
uchar blue  = color[0];
uchar green = color[1];
uchar red   = color[2];
```

### 4.4 设置像素

```cpp
img.at<uchar>(y, x) = 128;
img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);  // 绿色
```

## 5. 常见类型（Type）

OpenCV 的数据类型定义方式如下：

```
CV_<bit-depth>{U|S|F}C<channels>
```

| 宏定义        | 含义                | 说明          |
| ---------- | ----------------- | ----------- |
| `CV_8UC1`  | 8-bit 无符号，单通道     | 灰度图像常用      |
| `CV_8UC3`  | 8-bit 无符号，三通道     | 彩色图像（BGR）常用 |
| `CV_32FC1` | 32-bit float，单通道  | 浮点图、深度图     |
| `CV_64FC1` | 64-bit double，单通道 | 高精度计算       |


## 6. 图像区域（ROI）

从原图中获取一部分子图（不复制数据）：

```cpp
cv::Rect roi(50, 50, 100, 100);
cv::Mat sub_img = img(roi);  // 子图像
```


## 7. 克隆与复制

```cpp
cv::Mat A = img;        // 共享数据（引用计数+1）
cv::Mat B = img.clone(); // 完全复制（新的内存）
cv::Mat C;
img.copyTo(C);          // 完全复制（等价于 clone）
```


## 8. `Mat` 与其他容器互转

```cpp
std::vector<float> vec = {1,2,3,4,5,6};
cv::Mat mat(vec);  // 从 vector 创建 Mat

cv::Mat mat2(3, 1, CV_32F);
std::vector<float> out;
mat2.copyTo(out);  // 转回 vector
```

## 9. 图像访问方式
访问图像中像素值有多种方法，适用于不同的图像类型


首先需要知道图像的类型和通道数（`CV_8UC1`, `CV_8UC3`, `CV_32FC1` 等）：

```cpp
int type = img.type();         // 图像类型（整型数值）
int channels = img.channels(); // 通道数
```

| 方式           | 优点    | 适用场景   | 是否安全    | 是否推荐遍历中使用   |
| ------------ | ----- | ------ | ------- | ----------- |
| `at<T>(y,x)` | 简洁、安全 | 所有通道   | ✅ 安全    | ❌ 较慢（带边界检查） |
| `ptr<T>(y)`  | 高效、直接 | 所有通道   | ⚠ 不安全   | ✅ 推荐遍历使用    |
| 指针 + step    | 最高性能  | 高级场景   | ❌ 易错    | ✅ 高性能遍历     |
| 迭代器      | 安全可读  | 图像整体访问 | ✅ 安全 | ❌ 仅底层使用     |
| `data`       | 原始指针  | 图像整体访问 | ❌ 非常不安全 | ❌ 仅底层使用     |



### 9.1 `at<T>(y, x)` 

**单通道灰度图**

```cpp
cv::Mat gray = cv::imread("gray.jpg", cv::IMREAD_GRAYSCALE);
uchar pixel = gray.at<uchar>(y, x);      // 获取像素
gray.at<uchar>(y, x) = 128;              // 修改像素
```

**彩色图像（3通道）**

```cpp
cv::Mat img = cv::imread("color.jpg");
cv::Vec3b& color = img.at<cv::Vec3b>(y, x);  // BGR 顺序
uchar B = color[0];
uchar G = color[1];
uchar R = color[2];

color[2] = 255;  // 设置红色通道
```

**浮点图像**

```cpp
cv::Mat depth(480, 640, CV_32FC1);
float d = depth.at<float>(y, x);
```

---

### 9.2 `ptr<T>(y)[x]` （推荐遍历中使用）

`ptr`函数可以得到图像任意行的首地址

**单通道**

```cpp
for (int y = 0; y < gray.rows; ++y) {
    uchar* row_ptr = gray.ptr<uchar>(y);
    for (int x = 0; x < gray.cols; ++x) {
        uchar pixel = row_ptr[x];
        row_ptr[x] = 255 - pixel;
    }
}
```

**彩色图像（3 通道）**

```cpp
for (int y = 0; y < img.rows; ++y) {
    cv::Vec3b* row_ptr = img.ptr<cv::Vec3b>(y);
    for (int x = 0; x < img.cols; ++x) {
        row_ptr[x][0] = 255 - row_ptr[x][0];  // 反转蓝色通道
    }
}
```

---

### 9.3 使用原始指针和 `step`（高效）

> 用于最大性能优化，但代码较复杂

```cpp
uchar* base = img.data;
int step = img.step;  // 每行的字节数

for (int y = 0; y < img.rows; ++y) {
    uchar* row_ptr = base + y * step;
    for (int x = 0; x < img.cols; ++x) {
        uchar b = row_ptr[x * 3 + 0];
        uchar g = row_ptr[x * 3 + 1];
        uchar r = row_ptr[x * 3 + 2];
        // 操作...
    }
}
```

### 9.4 迭代器



**1. 灰度图反色（CV\_8UC1）**

```cpp
// 使用 Mat_<T>::iterator 遍历
    for (cv::Mat_<uchar>::iterator it = gray.begin(); it != gray.end(); ++it) {
        *it = 255 - *it;  // 像素反转
    }
```

**彩色图像（CV\_8UC3）**


```cpp
// 使用 Mat_<Vec3b>::iterator 遍历
    for (cv::Mat_<cv::Vec3b>::iterator it = color.begin(); it != color.end(); ++it) {
        (*it)[0] = 255 - (*it)[0];  // B
        (*it)[1] = 255 - (*it)[1];  // G
        (*it)[2] = 255 - (*it)[2];  // R
    }
```



### 9.5 常见错误避免

1. `at<T>` 中 `T` 与图像类型要匹配，如 `CV_8UC3` 用 `cv::Vec3b`，否则会崩溃。
2. 坐标顺序是 `y, x`，不是 `x, y`！
3. 不要混用 `data[]` 和 `ptr<>`。
4. 浮点图像用 `float` 或 `double`，不要用 `uchar` 取值。

### 9.6 完整示例

```cpp
cv::Mat img = cv::imread("color.jpg");
for (int y = 0; y < img.rows; ++y) {
    cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
    for (int x = 0; x < img.cols; ++x) {
        row[x][0] = 255 - row[x][0];  // B
        row[x][1] = 255 - row[x][1];  // G
        row[x][2] = 255 - row[x][2];  // R
    }
}
cv::imshow("Inverted", img);
cv::waitKey(0);
```


## demo

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        std::cerr << "读取失败\n";
        return -1;
    }

    std::cout << "图像大小: " << img.cols << " x " << img.rows << "\n";
    std::cout << "通道数: " << img.channels() << "\n";

    cv::Rect roi(50, 50, 100, 100);
    cv::Mat sub_img = img(roi);
    cv::imshow("ROI", sub_img);

    cv::Mat clone_img = img.clone();
    clone_img.setTo(cv::Scalar(0, 0, 255));  // 全图变红
    cv::imshow("Red", clone_img);

    cv::waitKey(0);
    return 0;
}
```



