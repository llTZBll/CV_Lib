import os
import sys

import cv2
import numpy as np


def imread_zh(path: str, flags: int = cv2.IMREAD_COLOR):
    """
    Windows 下对中文路径更稳的读取方式：
    用 numpy.fromfile 读取字节，再用 cv2.imdecode 解码。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_zh(path: str, img: np.ndarray) -> bool:
    """
    Windows 下对中文文件名更稳的保存方式：
    用 cv2.imencode 编码，再用 ndarray.tofile 写入。
    """
    try:
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".jpg"
            path = path + ext
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def _ensure_result_dir(result_dir: str) -> str:
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def _pick_input_image(user_path: str | None) -> str:
    if user_path:
        return user_path

    # 优先兼容作业文档的默认文件名；再兼容本仓库的数据集命名
    candidates = [
        os.path.join("answer_card.jpg"),
        os.path.join("DataSets", "answer_card", "answer_card.jpg"),
        os.path.join("DataSets", "answer_card", "1.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


def basic_operation(img_path: str, result_dir: str):
    """
    模块1：图像IO、属性分析、几何变换、色彩转换
    返回：img（原始图像）、img_cropped（裁剪后核心区域）、img_gray（灰度图）、img_hsv_h（HSV的H通道）
    """
    img = imread_zh(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("错误：无法读取答题卡图像，请检查文件路径或文件完整性！")
        print("img_path =", os.path.abspath(img_path))
        return None, None, None, None


    # 1.2 属性分析
    h, w, c = img.shape
    dtype = img.dtype
    print(f"答题卡图像属性 - 宽：{w}px，高：{h}px，通道数：{c}，数据类型：{dtype}")

    # 1.3.1 缩放 70%
    scale = 1.1
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    imwrite_zh(os.path.join(result_dir, "02-缩放后答题卡.jpg"), img_resized)


    h_r, w_r = img_resized.shape[:2]
    center = (w_r / 2.0, h_r / 2.0)

    # 手动旋转矫正角度（单位：度）
    # OpenCV：正角度=逆时针；负角度=顺时针
    # 你只需要改下面这一行数值来反复测试效果
    deskew_angle = 4.3

    rot_matrix = cv2.getRotationMatrix2D(center, deskew_angle, 1.0)

    cos = abs(rot_matrix[0, 0])
    sin = abs(rot_matrix[0, 1])
    new_w_r = int(h_r * sin + w_r * cos)
    new_h_r = int(h_r * cos + w_r * sin)
    rot_matrix[0, 2] += (new_w_r / 2.0) - center[0]
    rot_matrix[1, 2] += (new_h_r / 2.0) - center[1]

    img_rotated = cv2.warpAffine(
        img_resized,
        rot_matrix,
        (new_w_r, new_h_r),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    imwrite_zh(os.path.join(result_dir, "03-旋转矫正后.jpg"), img_rotated)

    # 1.3.3 裁剪核心区域（根据分隔线/条码区域自适应定位）
    h_rr, w_rr = img_rotated.shape[:2]
    gray_rr = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
    dark = (gray_rr < 90).astype(np.uint8)  # 黑线/涂黑选项/条码等

    row_sum = dark.sum(axis=1)
    col_sum = dark.sum(axis=0)

    # 1) 用“上半部分的粗横线”作为答题区起始（排除顶部抬头/准考证号等）
    y0, y1 = int(h_rr * 0.12), int(h_rr * 0.55)
    if y1 <= y0:
        y0, y1 = 0, h_rr
    y_sep = int(np.argmax(row_sum[y0:y1]) + y0)
    crop_h_start = min(max(y_sep + 8, 0), h_rr - 1)

    # 2) 右侧条码列通常“极黑且很高”，据此裁掉右边条码区域
    x0 = int(w_rr * 0.65)
    if x0 >= w_rr:
        x0 = 0
    barcode_cols = np.where(col_sum[x0:] > (h_rr * 0.35))[0]
    if barcode_cols.size > 0:
        crop_w_end = int(x0 + barcode_cols[0] - 8)
    else:
        crop_w_end = int(w_rr * 0.92)

    # 3) 左/下边界用内容的外接范围做收紧
    content_cols = np.where(col_sum > (h_rr * 0.02))[0]
    crop_w_start = int(max(content_cols[0] - 8, 0)) if content_cols.size > 0 else int(w_rr * 0.03)

    content_rows = np.where(row_sum > (w_rr * 0.02))[0]
    crop_h_end = int(min(content_rows[-1] + 8, h_rr)) if content_rows.size > 0 else int(h_rr * 0.95)

    # 4) 兜底与安全夹取
    crop_w_start = int(np.clip(crop_w_start, 0, w_rr - 2))
    crop_w_end = int(np.clip(crop_w_end, crop_w_start + 1, w_rr))
    crop_h_start = int(np.clip(crop_h_start, 0, h_rr - 2))
    crop_h_end = int(np.clip(crop_h_end, crop_h_start + 1, h_rr))

    img_cropped = img_rotated[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    imwrite_zh(os.path.join(result_dir, "04-裁剪后核心区域.jpg"), img_cropped)

    # 1.4 色彩空间转换
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    imwrite_zh(os.path.join(result_dir, "05-灰度答题卡.jpg"), img_gray)

    img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    img_hsv_h = img_hsv[:, :, 0]
    imwrite_zh(os.path.join(result_dir, "06-HSV_H通道.jpg"), img_hsv_h)

    return img, img_cropped, img_gray, img_hsv_h


def image_processing(img_gray: np.ndarray, result_dir: str):
    """
    模块2：滤波、阈值、形态学、边缘检测
    返回：img_gaussian、img_bilateral、img_adaptive、img_canny
    """
    img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1.2)
    imwrite_zh(os.path.join(result_dir, "07-高斯滤波降噪.jpg"), img_gaussian)

    img_median = cv2.medianBlur(img_gray, 5)
    imwrite_zh(os.path.join(result_dir, "08-中值滤波降噪.jpg"), img_median)

    img_bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
    imwrite_zh(os.path.join(result_dir, "09-双边滤波降噪.jpg"), img_bilateral)

    _, img_thresh = cv2.threshold(img_gaussian, 127, 255, cv2.THRESH_BINARY)
    imwrite_zh(os.path.join(result_dir, "10-全局二值化.jpg"), img_thresh)

    img_adaptive = cv2.adaptiveThreshold(
        img_gaussian,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    imwrite_zh(os.path.join(result_dir, "11-自适应阈值.jpg"), img_adaptive)

    kernel = np.ones((3, 3), np.uint8)

    img_erode = cv2.erode(img_adaptive, kernel, iterations=1)
    imwrite_zh(os.path.join(result_dir, "12-腐蚀处理.jpg"), img_erode)

    img_dilate = cv2.dilate(img_adaptive, kernel, iterations=1)
    imwrite_zh(os.path.join(result_dir, "13-膨胀处理.jpg"), img_dilate)

    img_open = cv2.morphologyEx(img_adaptive, cv2.MORPH_OPEN, kernel)
    imwrite_zh(os.path.join(result_dir, "14-开运算处理.jpg"), img_open)

    img_close = cv2.morphologyEx(img_adaptive, cv2.MORPH_CLOSE, kernel)
    imwrite_zh(os.path.join(result_dir, "15-闭运算处理.jpg"), img_close)

    img_canny = cv2.Canny(img_gaussian, 80, 180)
    imwrite_zh(os.path.join(result_dir, "16-Canny边缘检测.jpg"), img_canny)

    img_sobel_x = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    imwrite_zh(os.path.join(result_dir, "17-Sobel_X边缘.jpg"), img_sobel_x)

    img_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F, ksize=3)
    img_laplacian = cv2.convertScaleAbs(img_laplacian)
    imwrite_zh(os.path.join(result_dir, "18-Laplacian边缘.jpg"), img_laplacian)

    return img_gaussian, img_bilateral, img_adaptive, img_canny


def _rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a_x2, b_x2), min(a_y2, b_y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh - inter)
    return inter / union if union > 0 else 0.0


def _gen_answer_layout_85() -> list[dict]:
    """
    你只识别“这一张答题卡”，所以把布局参数全写死在这里，方便你直接改数值调参。

    返回 list，每一项是：
      {"q": 题号int, "opts": {"A":(x,y,w,h),"B":...,"C":...,"D":...}}

    重要：这些坐标是以 `img_cropped`（04-裁剪后核心区域）为坐标系。
    你第一次运行后，建议打开 `answer_card_result/21-答案框叠加.jpg` 看框位是否对齐。
    """
    # ===================== 你主要改这里（写死参数） =====================
    # 单个小方框尺寸（以 `04-裁剪后核心区域` 的像素为准）
    bubble_w = 28
    bubble_h = 10

    # 你给的“每行A框顶边y”（从上到下）
    row_a_y = [68, 232, 398, 563, 729]

    # 你给的“每5题组左边线x”（从左到右，对应 1-10 / 11-35 / 36-60 / 61-85 四列题块）
    block_left_x = [36, 283, 530, 776]

    # 同一组内 5 个题（1->2->3->4->5）的水平步长
    q_dx = 41

    # 你给的 block_left_x 若已是“第1题A框左边线”，这里应为 0；
    # 若以后你给的是“题块左边线”，再把它改成对应偏移即可。
    inner_dx = 0

    # 同一题 A/B/C/D 竖排的垂直步长
    opt_dy = 28
    # ================================================================

    def add_block_rows(start_q: int, block_x: int, rows_y: list[int], layout: list[dict]):
        for r, y_a in enumerate(rows_y):
            for c in range(5):
                q = start_q + r * 5 + c
                x_a = int(block_x + inner_dx + c * q_dx)
                y_a = int(y_a)
                opts = {
                    "A": (x_a, y_a + 0 * opt_dy, bubble_w, bubble_h),
                    "B": (x_a, y_a + 1 * opt_dy, bubble_w, bubble_h),
                    "C": (x_a, y_a + 2 * opt_dy, bubble_w, bubble_h),
                    "D": (x_a, y_a + 3 * opt_dy, bubble_w, bubble_h),
                }
                layout.append({"q": q, "opts": opts})

    layout: list[dict] = []
    # 左起第1列只包含 1-10（两行）
    add_block_rows(1, block_left_x[0], row_a_y[:2], layout)
    # 其余三列各 5 行
    add_block_rows(11, block_left_x[1], row_a_y, layout)
    add_block_rows(36, block_left_x[2], row_a_y, layout)
    add_block_rows(61, block_left_x[3], row_a_y, layout)
    return layout


def extract_answers_85(
    img_cropped_bgr: np.ndarray,
    img_bilateral_gray: np.ndarray,
    result_dir: str,
):

    # ===================== 识别参数 =====================
    bin_thresh = 165     # 二值化阈值（越大越“容易判黑”）
    morph_k = 3          # 形态学核大小
    morph_iter = 1       # 形态学迭代次数
    min_fill_ratio = 0.18  # 最低填涂占比，低于则判空

    # 若你想用“连通域bbox IoU”而不是占比，打开它（通常先用占比更好调）
    use_component_iou = False
    comp_min_area = 40
    comp_max_area = 2000
    iou_threshold = 0.12
    # ================================================================

    _, bw = cv2.threshold(img_bilateral_gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((morph_k, morph_k), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    imwrite_zh(os.path.join(result_dir, "21-答题卡二值_涂黑为白.jpg"), bw)

    layout = _gen_answer_layout_85()

    comp_bboxes: list[tuple[int, int, int, int]] = []
    if use_component_iou:
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for label in range(1, n):
            x, y, w, h, area = stats[label]
            if area < comp_min_area or area > comp_max_area:
                continue
            comp_bboxes.append((int(x), int(y), int(w), int(h)))

    vis = img_cropped_bgr.copy()
    answers: dict[int, str] = {}
    scores_debug: dict[int, dict[str, float]] = {}

    h_img, w_img = bw.shape[:2]

    # 可视化开关：
    # - 你现在希望第22张“框出所有选项填涂小区域”，所以默认打开框
    # - 若你之后还想看“5题组基准线”，把下面两个开关改一下即可
    VIS_DRAW_ALL_BUBBLES = True
    VIS_DRAW_GROUP_GUIDES = False

    if VIS_DRAW_GROUP_GUIDES:
        # 第22张：只绘制每个“5题组”的上边线和左边线（L形标线），用于校验你给的基准线是否正确
        guide_row_a_y = [68, 232, 398, 563, 729]
        guide_block_left_x = [36, 283, 530, 776]
        guide_rows_by_block = [
            guide_row_a_y[:2],  # 第1列只有 1-10（两行），中间是注意事项
            guide_row_a_y,
            guide_row_a_y,
            guide_row_a_y,
        ]
        for bi, bx in enumerate(guide_block_left_x):
            rows = guide_rows_by_block[bi]
            for ri, y in enumerate(rows):
                # 组宽：用相邻题块左边线推；最后一列用图像边界兜底
                if bi + 1 < len(guide_block_left_x):
                    gw = int(guide_block_left_x[bi + 1] - bx)
                else:
                    gw = int(w_img - bx - 3)
                # 组高：用相邻行y推；最后一行用图像边界兜底
                if ri + 1 < len(rows):
                    gh = int(rows[ri + 1] - y)
                else:
                    gh = int(h_img - y - 3)
                gw = int(np.clip(gw, 10, w_img - bx - 1))
                gh = int(np.clip(gh, 10, h_img - y - 1))

                p0 = (int(bx), int(y))
                p1 = (int(bx + gw), int(y))
                p2 = (int(bx), int(y + gh))
                cv2.line(vis, p0, p1, (0, 0, 255), 2)  # 上边线
                cv2.line(vis, p0, p2, (0, 0, 255), 2)  # 左边线

    for item in layout:
        q = int(item["q"])
        opts: dict[str, tuple[int, int, int, int]] = item["opts"]

        best_opt = ""
        best_score = -1.0
        per_opt: dict[str, float] = {}

        for opt, (x, y, w, h) in opts.items():
            # 安全裁剪，避免坐标调参时越界直接崩
            x1 = int(np.clip(x, 0, w_img - 1))
            y1 = int(np.clip(y, 0, h_img - 1))
            x2 = int(np.clip(x + w, x1 + 1, w_img))
            y2 = int(np.clip(y + h, y1 + 1, h_img))
            roi = bw[y1:y2, x1:x2]

            if use_component_iou:
                box = (x1, y1, x2 - x1, y2 - y1)
                score = 0.0
                for cb in comp_bboxes:
                    score = max(score, _rect_iou(box, cb))
            else:
                # 覆盖率（涂黑为白=255）
                score = float((roi > 0).mean())

            per_opt[opt] = score
            if score > best_score:
                best_score = score
                best_opt = opt

        if use_component_iou:
            chosen = best_opt if best_score >= iou_threshold else ""
        else:
            chosen = best_opt if best_score >= min_fill_ratio else ""

        answers[q] = chosen
        scores_debug[q] = per_opt

        if VIS_DRAW_ALL_BUBBLES:
            # 可视化：画框 + 标注得分最高的选项
            for opt, (x, y, w, h) in opts.items():
                color = (0, 255, 0) if opt == chosen and chosen != "" else (80, 80, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            # 题号标注在A框左上方
            xa, ya, _, _ = opts["A"]
            cv2.putText(
                vis,
                str(q),
                (xa, max(ya - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

    imwrite_zh(os.path.join(result_dir, "22-答案框叠加.jpg"), vis)

    # 输出（按题号排序）
    print("=" * 50)
    print("识别结果（85题）：")
    line = []
    for q in range(1, 86):
        ans = answers.get(q, "")
        token = f"{q}:{ans if ans else '_'}"
        line.append(token)
        if len(line) >= 20:
            print("  " + "  ".join(line))
            line = []
    if line:
        print("  " + "  ".join(line))
    print("=" * 50)

    # 需要你调参时，可以把每题每选项得分打印出来（默认不打印，太吵）
    # for q in range(1, 86):
    #     print(q, scores_debug.get(q, {}))


def interactive_pick_rois(img_bgr: np.ndarray, win_name: str = "pick"):
    """
    可选：用鼠标框选 ROI，按 Enter/Space 确认，按 Esc 结束。
    控制台会输出 (x,y,w,h) 方便你复制到 `_gen_answer_layout_85()` 里写死。

    用法（示例）：
      interactive_pick_rois(img_cropped)
    """
    show = img_bgr.copy()
    rois: list[tuple[int, int, int, int]] = []
    while True:
        r = cv2.selectROI(win_name, show, showCrosshair=True, fromCenter=False)
        x, y, w, h = map(int, r)
        if w <= 0 or h <= 0:
            break
        rois.append((x, y, w, h))
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"ROI {len(rois)} = ({x}, {y}, {w}, {h})")
    cv2.destroyWindow(win_name)
    return rois


def feature_detection(img_cropped: np.ndarray, img_gray: np.ndarray, img_canny: np.ndarray, result_dir: str):
    """
    模块3：轮廓检测与角点检测
    返回：contour_count, max_area
    """
    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = img_cropped.copy()
    cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 2)

    max_area = 0.0
    max_contour = None
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area > max_area:
            max_area = area
            max_contour = cnt

    if max_contour is not None:
        cv2.drawContours(img_contour, [max_contour], -1, (0, 255, 255), 4)

    imwrite_zh(os.path.join(result_dir, "19-轮廓检测标注.jpg"), img_contour)

    print(f"轮廓总数：{len(contours)}，最大轮廓面积：{round(max_area, 2)} 像素²")

    corners = cv2.goodFeaturesToTrack(img_gray, 60, 0.01, 10)
    if corners is None:
        print("角点检测失败：未检测到角点")
        corners = np.empty((0, 1, 2), dtype=np.float32)

    corners = corners.astype(np.int32)
    img_corner = img_cropped.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_corner, (int(x), int(y)), 3, (0, 255, 0), -1)

    imwrite_zh(os.path.join(result_dir, "20-角点检测标注.jpg"), img_corner)

    return len(contours), max_area


def main():
    result_dir = _ensure_result_dir("answer_card_result")

    user_img_path = sys.argv[1] if len(sys.argv) >= 2 else None
    img_path = _pick_input_image(user_img_path)

    print("img_path =", os.path.abspath(img_path))
    print("result_dir =", os.path.abspath(result_dir))

    img, img_cropped, img_gray, _ = basic_operation(img_path, result_dir)
    if img is None:
        sys.exit(1)

    img_gaussian, img_bilateral, img_adaptive, img_canny = image_processing(img_gray, result_dir)
    contour_count, max_area = feature_detection(img_cropped, img_gray, img_canny, result_dir)

    # 如果你想先手工框选几个基准框（比如每个题块第一题的A框），把它改成 True 再运行。
    ENABLE_ROI_PICKER = False
    if ENABLE_ROI_PICKER:
        interactive_pick_rois(img_cropped, win_name="pick_rois_on_cropped")

    extract_answers_85(img_cropped, img_bilateral, result_dir)

    print("=" * 50)
    print("所有必做模块处理完成！")
    print("处理结果已保存至：", os.path.abspath(result_dir))
    print(f"轮廓总数：{contour_count}，最大轮廓面积：{round(max_area, 2)} 像素²")
    print("=" * 50)


if __name__ == "__main__":
    main()

