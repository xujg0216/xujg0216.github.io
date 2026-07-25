# 图像拼接算法

>基于 OpenCV 提供的 图像拼接（Image Stitching）功能来合成多张图像，生成一张大尺寸的拼接图像，通常用于创建全景图像或大图像的拼接。它的主要流程包括图像加载、拼接、显示和保存

### 1. **图像加载 (`load_images`)**


```python

def load_images(image_paths):
    images = [cv2.imread(image_path) for image_path in image_paths]    
    return images
```
- 该函数接收一个图像路径列表 `image_paths`，然后使用 OpenCV 的 `cv2.imread()` 函数加载每张图像并存储在列表中。
- 每个路径对应一张图像，最终返回一个包含图像数据的列表。


### 2. **图像拼接 (`stitch_images`)**

```python
def stitch_images(images):
    # 创建一个图像拼接器     
    stitcher = cv2.Stitcher_create()     
    # 拼接图像     
    status, result = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:         
        print("拼接失败！")         
        return None     
    return result
```
- `cv2.Stitcher_create()`：创建一个图像拼接器对象，用于执行拼接操作。该拼接器会自动选择合适的拼接算法，通常使用基于特征匹配的算法（如 SIFT、SURF 或 ORB）来对齐图像。
    
- `stitcher.stitch(images)`：该方法接受一个图像列表 `images`，并将这些图像拼接成一张全景图像。它返回两个值：
    
    - `status`：拼接的状态。如果 `status` 是 `cv2.Stitcher_OK`，则表示拼接成功，否则表示拼接失败。
    - `result`：拼接后的图像。
- 如果拼接失败，返回 `None`。



完整代码如下：

```python
import cv2
import numpy as np
import os

# 读取图像
def load_images(image_paths):
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images

# 图像拼接函数
def stitch_images(images):
    # 创建一个图像拼接器
    stitcher = cv2.Stitcher_create()
    # 拼接图像
    status, result = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        print("拼接失败！")
        return None
    return result

# 显示图像
def display_image(image, window_name='Stitched Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 保存拼接后的图像
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# 主函数
def main():
    # 设置要拼接的图像路径
    input_root = '/home/xujg/code/image_pre_process/sample_case/case2/concat_input'
    imgs_list = os.listdir(input_root)
    image_paths = [os.path.join(input_root, img_list) for img_list in imgs_list]
    
    # 加载图像
    images = load_images(image_paths)
    
    # 拼接图像
    stitched_image = stitch_images(images)
    
    if stitched_image is not None:
        # 显示拼接后的图像
        display_image(stitched_image, 'Stitched Image')
        
        # 保存拼接后的图像
        save_image(stitched_image, 'stitched_output5.jpg')
    else:
        print("图像拼接失败！")

if __name__ == '__main__':
    main()

```



### 问题

- 图像拼接的效果受图像选择的结果，如果图像的尺度， 方向差距过大时，拼接的结果效果很差
- 在进行拼接之前， 对图像进行了重投影，统一了空间分辨率等。