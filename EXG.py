from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Disable the DecompressionBombError by setting MAX_IMAGE_PIXELS to None
Image.MAX_IMAGE_PIXELS = None

# 1. RGB格式讀取圖像
image = Image.open('/home/kingargroo/corn/jpgimg/xinmi.png').convert('RGB')

# 2. EXG index
image_np = np.array(image).astype(np.float32)
R = image_np[:, :, 0]
G = image_np[:, :, 1]
B = image_np[:, :, 2]
ExG = 2 * G - R - B

# NORMALIZE
ExG_norm = ((ExG - ExG.min()) / (ExG.max() - ExG.min()) * 255).astype(np.uint8)

plt.figure(figsize=(8,6))
plt.imshow(ExG_norm, cmap='gray')
plt.title('EXG Index')
plt.axis('off')
plt.savefig('exg.png')

# 3. Otsu
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

# 4. filtering
binary_image = Image.fromarray((binary * 255).astype(np.uint8))
binary_image = binary_image.filter(ImageFilter.MinFilter(3))  # 腐蚀
binary_image = binary_image.filter(ImageFilter.MaxFilter(3))  # 膨胀

binary_image.save('binary_image.png')

# 5. 檢測橫向過道
binary_np = np.array(binary_image)
horizontal_projection = np.sum(binary_np, axis=1)
threshold_projection = np.max(horizontal_projection) * 0.5  # 閾值可根據需要調整
rows_with_paths = np.where(horizontal_projection < threshold_projection)[0]
plt.figure(figsize=(10,5))
plt.plot(horizontal_projection)
plt.xlabel('Y')
plt.ylabel('sum of horizontal intensity')
plt.savefig('horizontal_projection.png')

# 合併相鄰的橫向過道
min_distance_h = 200
rows_with_paths_sorted = np.sort(rows_with_paths)
merged_rows = []

if len(rows_with_paths_sorted) > 0:
    current_cluster = [rows_with_paths_sorted[0]]
    for row in rows_with_paths_sorted[1:]:
        if row - current_cluster[-1] <= min_distance_h:
            current_cluster.append(row)
        else:
            merged_row = int(np.mean(current_cluster))
            merged_rows.append(merged_row)
            current_cluster = [row]
    merged_row = int(np.mean(current_cluster))
    merged_rows.append(merged_row)

# 6. 檢測縱向過道
vertical_projection = np.sum(binary_np, axis=0)
threshold_projection = np.max(vertical_projection) * 0.05  # 閾值可根據需要調整
columns_with_paths = np.where(vertical_projection < threshold_projection)[0]
plt.figure(figsize=(10,5))
plt.plot(vertical_projection)
plt.xlabel('X')
plt.ylabel('sum of vertical intensity')
plt.savefig('vertical_projection.png')

# 合併相鄰的縱向過道
columns_with_paths_sorted = np.sort(columns_with_paths)
merged_columns = []
min_distance_c=15
if len(columns_with_paths_sorted) > 0:
    current_cluster = [columns_with_paths_sorted[0]]
    for col in columns_with_paths_sorted[1:]:
        if col - current_cluster[-1] <= min_distance_c:
            current_cluster.append(col)
        else:
            merged_column = int(np.mean(current_cluster))
            merged_columns.append(merged_column)
            current_cluster = [col]
    merged_column = int(np.mean(current_cluster))
    merged_columns.append(merged_column)

# 畫出檢測到的橫向和縱向過道
detected_paths_image = image.copy()
draw = ImageDraw.Draw(detected_paths_image)

# 畫橫向過道
for row in merged_rows:
    draw.line([(0, row), (image.width, row)], fill='red', width=2)

# 畫縱向過道
for col in merged_columns:
    draw.line([(col, 0), (col, image.height)], fill='blue', width=2)

# 保存檢測到的過道圖像為PNG文件
detected_paths_image.save('detected_cross_paths_merged.png')
