OpenCV 综合实战作业：答题卡自动识别与预处理系统
一、应用背景
在学生考试、课堂测验场景中，答题卡批改是教师的常规工作，传统人工批改耗时久、易出错，且难以快速统计答题正确率。基于OpenCV的机器视觉技术，可实现答题卡的自动化预处理与特征提取，核心流程为：读取答题卡图像→几何矫正（解决倾斜）→降噪增强（提升清晰度）→边缘与轮廓检测（定位答题区域）→角点检测（校准答题卡坐标），为后续识别涂卡区域、自动判分提供高质量图像支撑。
本次作业模拟中小学标准化答题卡（客观题涂卡）的预处理场景，要求使用OpenCV完成全流程操作，掌握图像IO、几何变换、图像处理、特征检测的核心用法，贴合真实教学场景，易获取样张、易验证效果。
二、详细作业要求（全覆盖OpenCV三大核心模块，分模块、有标准、可验收）
核心目标
模拟答题卡自动预处理流程，通过OpenCV完成“图像读取→异常处理→几何矫正→色彩转换→降噪增强→阈值处理→形态学操作→边缘检测→轮廓提取→角点检测→结果保存”全链路操作，定位答题卡边框和涂卡区域，为后续自动判分奠定基础，同时全面掌握OpenCV核心函数的用法。
分模块详细要求（必做+选做，权重明确，验收标准清晰）
模块1：基础核心操作（必做，权重30%）
核心目标：完成答题卡图像的读取、属性分析、几何矫正和色彩转换，解决拍摄过程中的倾斜、尺寸不统一问题，为后续处理做准备。
| 序号 | 具体要求 | 技术要点 | 验收标准 |
| --- | --- | --- | --- |
| 1.1 | 图像IO操作（读取、展示、保存） | `cv2.imread()`、`cv2.imshow()`、`cv2.imwrite()`、异常处理 | 1) 读取指定路径的 `answer_card.jpg`，增加异常处理（路径错误、文件损坏、图像为空）<br>2) 若读取失败，打印清晰提示（如“错误：无法读取答题卡图像，请检查文件路径或文件完整性！”）并终止程序<br>3) 展示原始答题卡图像（窗口命名：「01-原始答题卡」），等待按键后关闭该窗口<br>4) 后续所有处理结果，统一保存到自动创建的 `answer_card_result` 文件夹中，文件名按要求命名，不允许遗漏 |
| 1.2 | 图像属性分析 | `img.shape`、`img.dtype` | 读取图像后，提取并打印图像核心属性：宽度（W）、高度（H）、通道数（C）、像素数据类型 |
| 1.3 | 图像几何变换（缩放、旋转、裁剪） | `cv2.resize()`、`cv2.getRotationMatrix2D()`、`cv2.warpAffine()`、切片裁剪 | 1) 缩放：将原始答题卡图像缩放到原尺寸的 70%（W×0.7，H×0.7），保存为 `02-缩放后答题卡.jpg`，保证图像不拉伸、不模糊<br>2) 旋转矫正：以缩放后图像的中心为旋转点，顺时针旋转 10°（模拟矫正拍摄倾斜的答题卡），保存为 `03-旋转矫正后.jpg`，旋转后不裁剪答题卡边框<br>3) 裁剪：从旋转矫正后的图像中，裁剪出答题卡核心区域（去除边缘空白，坐标范围：H/6~H×5/6，W/6~W×5/6），保存为 `04-裁剪后核心区域.jpg`，确保裁剪后包含完整的涂卡区域和答题卡边框<br>4) 所有几何变换函数参数设置合理，无明显失真 |
| 1.4 | 色彩空间转换 | `cv2.cvtColor()` | 1) 将裁剪后的核心区域图像转换为灰度图（用于后续滤波、阈值处理），保存为 `05-灰度答题卡.jpg`<br>2) 将裁剪后的核心区域图像转换为 HSV 色彩空间，提取 H 通道（用于区分答题卡与背景），保存为 `06-HSV_H通道.jpg`<br>3) 转换后的图像清晰，无颜色失真（灰度图无杂色，HSV 通道图能区分答题卡区域） |
模块2：图像处理操作（必做，权重40%）
核心目标：对灰度答题卡进行降噪、增强、形态学优化，突出答题卡边框和涂卡区域，去除拍摄过程中的噪点、光照不均等问题，为特征检测做准备。
| 序号 | 具体要求 | 技术要点 | 验收标准 |
| --- | --- | --- | --- |
| 2.1 | 图像滤波降噪（3种滤波方式） | `cv2.GaussianBlur()`、`cv2.medianBlur()`、`cv2.bilateralFilter()` | 1) 以模块1生成的「灰度答题卡.jpg」为处理对象，分别执行 3 种滤波操作：<br>- 高斯滤波：核大小 5×5，σ=1.2（去除轻微噪点），保存为 `07-高斯滤波降噪.jpg`<br>- 中值滤波：核大小 5×5（去除椒盐噪点，模拟拍摄时的杂点），保存为 `08-中值滤波降噪.jpg`<br>- 双边滤波：d=9，σColor=75，σSpace=75（保留答题卡边框、涂卡框边缘，同时降噪），保存为 `09-双边滤波降噪.jpg`<br>2) 滤波结果需满足：降噪明显，且不模糊答题卡边框和涂卡区域，能清晰区分图像细节 |
| 2.2 | 阈值处理（2种阈值方式） | `cv2.threshold()`、`cv2.adaptiveThreshold()` | 1) 以「高斯滤波降噪.jpg」为处理对象，执行 2 种阈值操作，将灰度图转换为二值图（突出答题卡前景，抑制背景）：<br>- 全局二值化：阈值 127，最大值 255，阈值类型为 `cv2.THRESH_BINARY`，保存为 `10-全局二值化.jpg`<br>- 自适应阈值：块大小 11，常数 2，阈值类型为 `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`（处理拍摄时的光照不均问题），保存为 `11-自适应阈值.jpg`<br>2) 二值化结果需满足：答题卡边框、涂卡框为黑色（或白色），背景为白色（或黑色），无大面积噪点，涂卡框轮廓清晰可辨 |
| 2.3 | 形态学操作（4种操作） | `cv2.erode()`、`cv2.dilate()`、`cv2.morphologyEx()`（开运算、闭运算） | 1) 以「自适应阈值.jpg」为处理对象，使用 3×3 的矩形结构元素（kernel），执行 4 种形态学操作：<br>- 腐蚀：迭代次数 1（去除二值图中的细小噪点），保存为 `12-腐蚀处理.jpg`<br>- 膨胀：迭代次数 1（填充涂卡框内部的细小缝隙），保存为 `13-膨胀处理.jpg`<br>- 开运算：先腐蚀后膨胀（去除涂卡框周围的小杂点，不破坏涂卡框轮廓），保存为 `14-开运算处理.jpg`<br>- 闭运算：先膨胀后腐蚀（填充涂卡框内部的小空洞，使轮廓更完整），保存为 `15-闭运算处理.jpg`<br>2) 形态学操作结果需满足：去除噪点、填充缝隙，答题卡边框和涂卡框轮廓更清晰，无明显变形 |
| 2.4 | 边缘检测（3种算法） | `cv2.Canny()`、`cv2.Sobel()`、`cv2.Laplacian()` | 1) 以「高斯滤波降噪.jpg」为处理对象，执行 3 种边缘检测算法，提取答题卡边框和涂卡框的边缘：<br>- Canny 边缘检测：阈值 80/180（最常用，边缘检测效果最佳），保存为 `16-Canny边缘检测.jpg`<br>- Sobel 算子：X 方向，核大小 3×3，转换为 uint8 类型，保存为 `17-Sobel_X边缘.jpg`<br>- Laplacian 算子：核大小 3×3，转换为 uint8 类型，保存为 `18-Laplacian边缘.jpg`<br>2) 边缘检测结果需满足：能完整勾勒出答题卡边框和所有涂卡框的边缘，无明显断裂、无过多杂边 |
模块3：特征检测操作（必做，权重20%）
核心目标：提取答题卡的轮廓和角点，定位答题卡边框（最大轮廓）和涂卡区域（细小轮廓），校准答题卡坐标，为后续识别涂卡状态、自动判分提供依据。
| 序号 | 具体要求 | 技术要点 | 验收标准 |
| --- | --- | --- | --- |
| 3.1 | 轮廓检测与标注 | `cv2.findContours()`、`cv2.drawContours()`、`cv2.contourArea()` | 1) 以「Canny边缘检测.jpg」为处理对象，查找图像的外部轮廓（`cv2.RETR_EXTERNAL`），使用轮廓逼近算法（`cv2.CHAIN_APPROX_SIMPLE`）<br>2) 在模块1生成的「裁剪后核心区域.jpg」上绘制所有轮廓（红色，线宽 2），突出所有涂卡框和答题卡边框<br>3) 找出所有轮廓中面积最大的轮廓（即答题卡边框），用黄色粗线（线宽 4）标注，区分于其他细小轮廓（涂卡框）<br>4) 保存绘制后的图像为 `19-轮廓检测标注.jpg`<br>5) 在控制台打印 2 条信息：轮廓总数（涂卡框+答题卡边框）、最大轮廓面积（保留 2 位小数），示例：`轮廓总数：52，最大轮廓面积：45689.78 像素²`<br>6) 轮廓检测准确，无遗漏主要轮廓，最大轮廓能精准对应答题卡边框 |
| 3.2 | 角点检测与标注 | `cv2.goodFeaturesToTrack()`（Shi-Tomasi 角点检测） | 1) 以模块1生成的「灰度答题卡.jpg」为处理对象，执行 Shi-Tomasi 角点检测<br>2) 检测参数：最多检测 60 个角点，质量等级 0.01，角点间最小距离 10<br>3) 在模块1生成的「裁剪后核心区域.jpg」上绘制检测到的角点（绿色圆点，半径 3，填充），重点标注答题卡边框的四个角和涂卡框的拐角<br>4) 保存绘制后的图像为 `20-角点检测标注.jpg`<br>5) 角点检测准确，角点集中在答题卡边框、涂卡框拐角处，无过多无效角点（如背景区域的杂点） |
提交要求（规范统一，适合交作业）
1.代码文件：命名为 answer_card_process.py，代码结构清晰，包含完整注释（函数说明、关键步骤注释、参数说明），无语法错误，可直接运行；
2.结果文件夹：命名为 answer_card_result，包含所有必做模块要求保存的20张处理后图片（文件名严格按要求命名，无遗漏、无错误）；若完成选做模块，需额外包含保存的视频文件和截图帧；
3.运行日志：截图或文本文件（命名为 运行日志.txt），记录程序运行时控制台打印的信息（图像属性、轮廓总数、最大轮廓面积），以及程序运行过程中的关键截图（如原始图像窗口、轮廓检测窗口）；
4.样张文件：将使用的 answer_card.jpg 一并提交，便于教师验证代码运行效果。
四、完整例程代码（可直接运行，注释详细，适配所有作业要求）

```python
import cv2
import numpy as np
import os

# ====================== 初始化配置（无需修改，直接使用） ======================
# 1. 创建结果文件夹（自动创建，无需手动操作）
result_dir = "answer_card_result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 2. 定义答题卡样张路径（确保同目录下有 answer_card.jpg）
img_path = "answer_card.jpg"

# ====================== 模块1：基础核心操作（必做） ======================
def basic_operation(img_path):
    """
    基础核心操作：图像IO、属性分析、几何变换、色彩转换
    参数：img_path - 答题卡样张路径
    返回：img（原始图像）、img_cropped（裁剪后核心区域）、img_gray（灰度图）、img_hsv_h（HSV的H通道）
    """
    # 1.1 图像IO操作 + 异常处理
    img = cv2.imread(img_path)
    if img is None:
        print("错误：无法读取答题卡图像，请检查文件路径或文件完整性！")
        return None, None, None, None
    
    # 展示原始图像，等待按键后关闭
    cv2.imshow("01-原始答题卡", img)
    cv2.waitKey(0)  # 等待任意按键
    cv2.destroyWindow("01-原始答题卡")  # 关闭当前窗口
    
    # 1.2 图像属性分析
    h, w, c = img.shape  # 高度、宽度、通道数
    dtype = img.dtype    # 像素数据类型
    print(f"答题卡图像属性 - 宽：{w}px，高：{h}px，通道数：{c}，数据类型：{dtype}")
    
    # 1.3 几何变换（缩放、旋转、裁剪）
    # 1.3.1 缩放：原尺寸的70%
    scale = 0.7
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imwrite(os.path.join(result_dir, "02-缩放后答题卡.jpg"), img_resized)
    
    # 1.3.2 旋转矫正：顺时针旋转10°（中心旋转，不裁剪）
    h_r, w_r = img_resized.shape[:2]  # 缩放后图像的高度、宽度
    center = (w_r // 2, h_r // 2)     # 旋转中心（图像中心）
    # 旋转矩阵：center（旋转中心）、angle（旋转角度，顺时针为负，逆时针为正）、scale（缩放比例）
    rot_matrix = cv2.getRotationMatrix2D(center, -10, 1.0)
    img_rotated = cv2.warpAffine(img_resized, rot_matrix, (w_r, h_r))
    cv2.imwrite(os.path.join(result_dir, "03-旋转矫正后.jpg"), img_rotated)
    
    # 1.3.3 裁剪：去除边缘空白，保留核心区域
    crop_h_start = int(h_r / 6)  # 裁剪高度起始坐标
    crop_h_end = int(h_r * 5 / 6)# 裁剪高度结束坐标
    crop_w_start = int(w_r / 6)  # 裁剪宽度起始坐标
    crop_w_end = int(w_r * 5 / 6)# 裁剪宽度结束坐标
    img_cropped = img_rotated[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    cv2.imwrite(os.path.join(result_dir, "04-裁剪后核心区域.jpg"), img_cropped)
    
    # 1.4 色彩空间转换
    # 1.4.1 彩色转灰度
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(result_dir, "05-灰度答题卡.jpg"), img_gray)
    
    # 1.4.2 彩色转HSV，提取H通道
    img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    img_hsv_h = img_hsv[:, :, 0]  # H通道（索引0）
    cv2.imwrite(os.path.join(result_dir, "06-HSV_H通道.jpg"), img_hsv_h)
    
    return img, img_cropped, img_gray, img_hsv_h

# ====================== 模块2：图像处理操作（必做） ======================
def image_processing(img_gray):
    """
    图像处理操作：滤波降噪、阈值处理、形态学操作、边缘检测
    参数：img_gray - 模块1生成的灰度答题卡图像
    返回：img_gaussian（高斯滤波图）、img_adaptive（自适应阈值图）、img_canny（Canny边缘图）
    """
    # 2.1 滤波降噪（3种方式）
    # 2.1.1 高斯滤波
    img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1.2)
    cv2.imwrite(os.path.join(result_dir, "07-高斯滤波降噪.jpg"), img_gaussian)
    
    # 2.1.2 中值滤波
    img_median = cv2.medianBlur(img_gray, 5)
    cv2.imwrite(os.path.join(result_dir, "08-中值滤波降噪.jpg"), img_median)
    
    # 2.1.3 双边滤波
    img_bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
    cv2.imwrite(os.path.join(result_dir, "09-双边滤波降噪.jpg"), img_bilateral)
    
    # 2.2 阈值处理（2种方式）
    # 2.2.1 全局二值化
    _, img_thresh = cv2.threshold(img_gaussian, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(result_dir, "10-全局二值化.jpg"), img_thresh)
    
    # 2.2.2 自适应阈值（处理光照不均）
    img_adaptive = cv2.adaptiveThreshold(
        img_gaussian, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权自适应阈值
        cv2.THRESH_BINARY, 11, 2         # 块大小11，常数2
    )
    cv2.imwrite(os.path.join(result_dir, "11-自适应阈值.jpg"), img_adaptive)
    
    # 2.3 形态学操作（4种方式）
    kernel = np.ones((3, 3), np.uint8)  # 3×3矩形结构元素
    
    # 2.3.1 腐蚀
    img_erode = cv2.erode(img_adaptive, kernel, iterations=1)
    cv2.imwrite(os.path.join(result_dir, "12-腐蚀处理.jpg"), img_erode)
    
    # 2.3.2 膨胀
    img_dilate = cv2.dilate(img_adaptive, kernel, iterations=1)
    cv2.imwrite(os.path.join(result_dir, "13-膨胀处理.jpg"), img_dilate)
    
    # 2.3.3 开运算（先腐蚀后膨胀）
    img_open = cv2.morphologyEx(img_adaptive, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(result_dir, "14-开运算处理.jpg"), img_open)
    
    # 2.3.4 闭运算（先膨胀后腐蚀）
    img_close = cv2.morphologyEx(img_adaptive, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(result_dir, "15-闭运算处理.jpg"), img_close)
    
    # 2.4 边缘检测（3种方式）
    # 2.4.1 Canny边缘检测（重点）
    img_canny = cv2.Canny(img_gaussian, 80, 180)
    cv2.imwrite(os.path.join(result_dir, "16-Canny边缘检测.jpg"), img_canny)
    
    # 2.4.2 Sobel X边缘检测
    img_sobel_x = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)  # 转换为uint8类型，避免负数值
    cv2.imwrite(os.path.join(result_dir, "17-Sobel_X边缘.jpg"), img_sobel_x)
    
    # 2.4.3 Laplacian边缘检测
    img_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F, ksize=3)
    img_laplacian = cv2.convertScaleAbs(img_laplacian)  # 转换为uint8类型
    cv2.imwrite(os.path.join(result_dir, "18-Laplacian边缘.jpg"), img_laplacian)
    
    return img_gaussian, img_adaptive, img_canny

# ====================== 模块3：特征检测操作（必做） ======================
def feature_detection(img_cropped, img_gray, img_canny):
    """
    特征检测操作：轮廓检测、角点检测
    参数：img_cropped（裁剪后核心区域）、img_gray（灰度图）、img_canny（Canny边缘图）
    """
    # 3.1 轮廓检测与标注
    # 查找外部轮廓（只保留最外层轮廓，去除内部细小轮廓的嵌套）
    contours, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 绘制所有轮廓（红色，线宽2）
    img_contour = img_cropped.copy()  # 复制裁剪后的图像，避免修改原图
    cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 2)  # -1表示绘制所有轮廓
    
    # 查找最大轮廓（答题卡边框）
    max_area = 0
    max_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 计算每个轮廓的面积
        if area > max_area:
            max_area = area
            max_contour = cnt
    
    # 标注最大轮廓（黄色，线宽4）
    if max_contour is not None:
        cv2.drawContours(img_contour, [max_contour], -1, (0, 255, 255), 4)
    
    # 保存轮廓检测结果
    cv2.imwrite(os.path.join(result_dir, "19-轮廓检测标注.jpg"), img_contour)
    
    # 打印轮廓信息
    print(f"轮廓总数：{len(contours)}")
    print(f"最大轮廓面积：{round(max_area, 2)} 像素²")
    
    # 3.2 角点检测（Shi-Tomasi）与标注
    # 检测角点：最多60个，质量等级0.01，最小距离10
    corners = cv2.goodFeaturesToTrack(img_gray, 60, 0.01, 10)
    corners = np.int0(corners)  # 转换为整数坐标（角点坐标为浮点数，需转换）
    
    # 绘制角点（绿色圆点，半径3，填充）
    img_corner = img_cropped.copy()
    for corner in corners:
        x, y = corner.ravel()  # 将二维坐标转换为一维（x,y）
        cv2.circle(img_corner, (x, y), 3, (0, 255, 0), -1)  # -1表示填充圆点
    
    # 保存角点检测结果
    cv2.imwrite(os.path.join(result_dir, "20-角点检测标注.jpg"), img_corner)

# ====================== 主程序（程序入口，无需修改） ======================
if __name__ == "__main__":
    # 1. 执行模块1：基础核心操作
    img, img_cropped, img_gray, _ = basic_operation(img_path)
    if img is None:  # 若图像读取失败，终止程序
        exit()
    
    # 2. 执行模块2：图像处理操作
    img_gaussian, img_adaptive, img_canny = image_processing(img_gray)
    
    # 3. 执行模块3：特征检测操作
    feature_detection(img_cropped, img_gray, img_canny)
    

    
    # 提示处理完成
    print("="*50)
    print("所有必做模块处理完成！")
    print(f"处理结果已保存至：{os.path.abspath(result_dir)}")
    print("请按提交要求，整理代码、结果文件夹、运行日志和样张后提交作业。")
    print("="*50)

```