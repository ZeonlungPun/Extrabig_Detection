import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from transform_streamline import ImageTransform
"""
ExG指數加Otsu二指化，利用4個點提取出ROI；4個點對二值化圖像矯正；得到橫過道位置；映射回原來的圖像；提取每個Range區域；
依次矯正；保存圖像
"""
# 1. 讀取圖像
Image.MAX_IMAGE_PIXELS = None
image = Image.open('/home/kingargroo/corn/jpgimg/xinmi.tif').convert('RGB')

# 2. 讀取4個關鍵點
point_path = '/home/kingargroo/arieal/big/xinmi_points.csv'
point_df = pd.read_csv(point_path, header=None)
point_array = np.array(point_df, dtype=int)


if point_array.shape != (4, 2):
    raise ValueError("点数组应具有 (4, 2) 的形状，表示四个 (x, y) 坐标。")


polygon = [tuple(point) for point in point_array]
# 3. 創建mask
mask = Image.new('L', image.size, 0)
ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)

# 4. 應用mask提取ROI，其他區域去除
background = Image.new('RGB', image.size, (0, 0, 0))
mask_image = Image.composite(image, background, mask)

# 5. EXG index and Otsu 區分背景和前景
image_np = np.array(mask_image).astype(np.float32)
R = image_np[:, :, 0]
G = image_np[:, :, 1]
B = image_np[:, :, 2]
ExG = 2 * G - R - B

# NORMALIZE
ExG_norm = ((ExG - ExG.min()) / (ExG.max() - ExG.min()) * 255).astype(np.uint8)

# Otsu
def otsu_threshold(image):
    """
    最大化類間方差(前景和背景之間的方差)來選擇最佳的分割閾值
    """
#將圖像展平並計算每個像素值（0-255）的出現次數。
# 會返回一個長度為 256 的數組，每個元素代表該像素值在圖像中出現的次數。
    pixel_counts = np.bincount(image.flatten(), minlength=256)
#圖像的總像素數量，即圖像的大小
    total = image.size
#圖像中所有像素值的加權和，這是基於每個像素值與其出現頻率的加權平均。
    sum_total = np.dot(np.arange(256), pixel_counts)
#sumB：背景像素的總和。
#wB：背景的總像素權重（即背景的像素數量）。
#maximum：用於存儲最大類間方差。
#threshold：最終返回的最佳閾值。
    sumB, wB, maximum = 0.0, 0.0, 0.0
    threshold = 0
#在每次迭代中計算不同閾值 i 的類間方差
    for i in range(256):
#wB：背景的像素總權重（總數量），隨著閾值 i 增加，每次更新背景像素的數量。
        wB += pixel_counts[i]
        if wB == 0:
            continue
#wF：前景的像素總權重（總數量），等於圖像總像素數量減去背景像素數量。
        wF = total - wB
        if wF == 0:
            break
#sumB：背景的像素加權總和，隨著每次增加閾值的 i 更新
        sumB += i * pixel_counts[i]
#mB：背景的平均灰度值，通過將 sumB 除以 wB 計算。
        mB = sumB / wB
#mF：前景的平均灰度值，通過將 (sum_total - sumB) 除以 wF 計算。
        mF = (sum_total - sumB) / wF
#between：計算當前閾值下的類間方差
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold = i
    return threshold

threshold = otsu_threshold(ExG_norm)
binary = ExG_norm > threshold

# 6，filtering
binary_image = Image.fromarray((binary * 255).astype(np.uint8))
binary_image = binary_image.filter(ImageFilter.MinFilter(3))  # 腐蚀
binary_image = binary_image.filter(ImageFilter.MaxFilter(3))  # 膨胀

# 7，rectify the binary_image
dot_img=cv2.cvtColor(np.array(binary_image), cv2.COLOR_RGB2BGR)
out_path='./'
trans = ImageTransform(src_img=dot_img, input_points=point_array, transform_category='Perspective',
                       save=True, save_path=out_path)
roi,mat = trans.main()
binary_np = np.array(roi)

# 8, find the horizontal aisle (range seperation)
horizontal_projection = np.sum(binary_np, axis=1)
threshold_projection = np.max(horizontal_projection) * 0.5
rows_with_paths = np.where(horizontal_projection < threshold_projection)[0]
plt.figure(figsize=(10,5))
plt.plot(horizontal_projection)
plt.xlabel('Y')
plt.ylabel('sum of horizontal intensity')
plt.savefig('projection.png')

# 9. 合併冗餘
min_distance = 200
rows_with_paths_sorted = np.sort(rows_with_paths)
merged_rows = []

if len(rows_with_paths_sorted) > 0:
    current_cluster = [rows_with_paths_sorted[0]]
    for row in rows_with_paths_sorted[1:]:
        if row - current_cluster[-1] <= min_distance:
            current_cluster.append(row)
        else:
            # 计算當前簇的平均行索引
            merged_row = int(np.mean(current_cluster))
            merged_rows.append(merged_row)
            current_cluster = [row]
    # 添加最後一簇的平均行索引
    merged_row = int(np.mean(current_cluster))
    merged_rows.append(merged_row)


def map_line_back_to_original(mat, line_points):
    # 計算逆矩陣
    inv_mat = np.linalg.inv(mat)

    # 準備線條點 (假設 line_points 為 [[x1, y1], [x2, y2]])
    line_points = np.array([line_points], dtype='float32')

    # 使用逆矩陣將線條點映射回原圖
    original_points = cv2.perspectiveTransform(line_points, inv_mat)

    return original_points[0]


# 10，矯正圖畫線條和原圖畫斜線
roi=Image.fromarray(roi)
detected_paths_image = roi.copy()
draw1 = ImageDraw.Draw(detected_paths_image)

original_image=mask_image
# 將圖像轉換為 numpy array
image_array = np.array(original_image)
# 檢查是否需要從 RGB 轉換為 BGR（這樣才能避免顏色偏差）
if image_array.shape[-1] == 3:  # 確認是彩色圖像
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
original_image_=original_image.copy()
draw2 = ImageDraw.Draw(original_image_)

new_merged_rows=[]
for row in merged_rows:
    now_line_points=[[0, row],[roi.width, row]]
    #把矯正圖中檢測到的值線轉換回原圖，利用透視變換逆矩陣
    original_points=map_line_back_to_original(mat, now_line_points)
    draw1.line([(0, row), (roi.width, row)], fill='red', width=2)
    draw2.line([(original_points[0][0],original_points[0][1]), (original_points[1][0],original_points[1][1])], fill='red', width=8)
    new_merged_rows.append(original_points)

# 保存
detected_paths_image.save('detected_cross_paths_merged.png')
original_image_.save('detected_cross_paths_merged2.png')

#11，在原圖中提取每個range並保存下來

crop_areas = []
for i in range(len(new_merged_rows) - 1):
    # 取得相鄰兩條直線的端點
    line_top = new_merged_rows[i]
    line_bottom = new_merged_rows[i + 1]

    # 設定該區域的四個頂點（左上、右上、右下、左下）
    area_points = np.array([
        [line_top[0][0], line_top[0][1]],
        [line_top[1][0], line_top[1][1]],
        [line_bottom[1][0], line_bottom[1][1]],
        [line_bottom[0][0], line_bottom[0][1]]
    ], dtype=np.float32)

    # 創建遮罩以便從原圖中裁切出區域
    mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [area_points.astype(np.int32)], 255)

    # 使用遮罩提取區域，確保顏色保持原始
    cropped_area = cv2.bitwise_and(image_array, image_array, mask=mask)

    # 找到 bounding box 以裁切出最小矩形
    x, y, w, h = cv2.boundingRect(area_points)
    cropped_region = cropped_area[y:y + h, x:x + w]

    # 使用 area_points 和 目標矩形點計算透視變換矩陣
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(area_points, dst_points)

    # 對裁切的區域進行透視變換矯正
    transformed_region = cv2.warpPerspective(cropped_region, M, (w, h))

    # 保存透視變換後的矯正區域
    output_path = f'crop_area_corrected_{i}.png'
    cv2.imwrite(output_path, transformed_region)
    print(f'Saved corrected cropped area to {output_path}')



