import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
def SegemtPlot(img):
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[0:2]

    horizontal_lines = []
    h_ = 1
    # add first horizontal lines
    horizontal_lines.append(h_)

    while True:
        h_ += 280
        if h_ + 280 > h:
            break
        # find the minimun value in 100 pixel range
        pixel_list, index_list = [], []
        for i in range(280):
            total_pixel = np.sum(img[(h_ + i):(h_ + i + 5), :])
            pixel_list.append(total_pixel)
            index_list.append(h_ + i)
        min_index = np.argmin(pixel_list)
        min_h = index_list[min_index]
        if min_h - horizontal_lines[-1] < 280:
            continue
        horizontal_lines.append(min_h + 8)
    # add last horizontal lines
    horizontal_lines.append(h - 5)
    if horizontal_lines[-1] - horizontal_lines[-2] < 250:
        horizontal_lines.remove(horizontal_lines[-2])

    print('h:', horizontal_lines)

    w_ = 1
    vertical_lines = []
    vertical_lines.append(w_)

    while True:
        w_ += 20
        if w_ + 20 > w:
            break
        pixel_list, index_list = [], []
        for i in range(20):
            total_pixel = np.sum(img[:, (w_ + i):(w_ + i + 5)])
            pixel_list.append(total_pixel)
            index_list.append(w_ + i)
        min_index = np.argmin(pixel_list)
        min_w = index_list[min_index]
        if min_w - vertical_lines[-1] < 15:
            continue
        vertical_lines.append(min_w + 3)
    vertical_lines.append(w - 2)
    print('v:', vertical_lines)

    return horizontal_lines, vertical_lines


def check_plot(vertical_lines, img):
    new_vertical_lines = []
    ave_value = np.mean(vertical_lines)
    print("ave:", ave_value)
    for i in range(len(vertical_lines) - 1):
        start = vertical_lines[i]
        end = vertical_lines[i + 1]
        total_pixel = np.sum(img[:, start:end])
        if total_pixel < ave_value:
            continue
        else:
            new_vertical_lines.append(start)

    new_vertical_lines.append(vertical_lines[-1])
    # new_vertical_lines[0]=new_vertical_lines[0]+5
    return new_vertical_lines
def set_surrounding_pixels_to_zero(image, center_pixel):
    height, width = image.shape[:2]
    x, y = center_pixel

    # 檢查中心像素是否在圖像範圍內
    if x < 0 or y < 0 or x >= height or y >= width:
        return image

    # 確保周圍16個像素不超出圖像範圍
    x_start = max(0, x - 2)
    x_end = min(x + 3, height)
    y_start = max(0, y - 2)
    y_end = min(y + 3, width)

    # 將周圍16個像素設定為0
    image[x_start:x_end, y_start:y_end] = 0

    return image
def count_yield(image):
    num=0
    while True:
        white_pixels = np.argwhere(image != 0)
        for pixel in white_pixels:
            if image[pixel[0],pixel[1]]!=0:
                num+=1
                image = set_surrounding_pixels_to_zero(image, (pixel[0],pixel[1]))
                break
        if len(white_pixels)==0:
            break
    return num


def Visualize(save_path, vertical_lines, horizontal_lines, img):
    # visualize the line to check
    vertical_lines, horizontal_lines = np.array(vertical_lines), np.array(horizontal_lines)
    vertical_lines = vertical_lines[vertical_lines.argsort()]
    horizontal_lines = horizontal_lines[horizontal_lines.argsort()]

    plot = np.array([0, 0, 0, 0]).reshape((1, -1))

    sign_num = 0
    for index, i in enumerate(range(vertical_lines.shape[0] - 1)):
        if index % 2 == 0:
            sign_num += 1
        x1 = vertical_lines[i]
        x2 = vertical_lines[i + 1]

        sign_num_h = horizontal_lines.shape[0]
        for j in range(horizontal_lines.shape[0] - 1):
            sign_num_h -= 1
            y1 = horizontal_lines[j]
            y2 = horizontal_lines[j + 1]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, thickness=1)
            plot_ = np.array([x1, y1, x2, y2]).reshape((1, -1))
            plot = np.concatenate([plot, plot_], axis=0)
            if index % 2 == 0:
                cv2.putText(img, str(sign_num_h) + '-' + str(sign_num), (x1 + 2, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            255, thickness=1)
    plot = plot[1::]
    print(plot)
    cv2.imwrite(save_path, img)
    return plot



im_path='../kaiyuan/kaiyuanZQ15-1.png'
im_pil = Image.open(im_path)
im_cv2 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
rawimg=cv2.imread('../kaiyuan/kaiyuanZQ15-1_point.png',0)
horizontal_lines, vertical_lines = SegemtPlot(rawimg)
vertical_lines =check_plot(vertical_lines,rawimg)
save_path='../kaiyuan/plot_result.png'
Visualize(save_path, vertical_lines, horizontal_lines, rawimg)
 # visualize the line to check
vertical_lines, horizontal_lines = np.array(vertical_lines), np.array(horizontal_lines)
vertical_lines = vertical_lines[vertical_lines.argsort()]
horizontal_lines = horizontal_lines[horizontal_lines.argsort()]
plot = np.array([0, 0, 0, 0]).reshape((1, -1))
sign_num = 0
for index,i in enumerate(range(vertical_lines.shape[0] - 1)):
    if index%2==0:
        sign_num +=1
    x1 = vertical_lines[i]
    x2 = vertical_lines[i + 1]
    x1_,x2_=x1*4,x2*4

    sign_num_h=horizontal_lines.shape[0]
    for j in range(horizontal_lines.shape[0]-1):
        sign_num_h-=1
        y1 = horizontal_lines[j]
        y2 = horizontal_lines[j + 1]
        y1_,y2_=y1*4,y2*4
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
        w = x2 - x1
        h = y2 - y1
        cv2.rectangle(im_cv2, (x1_, y1_), (x2_, y2_), (0,0,255), thickness=3)
        plot_ = np.array([x1, y1, x2, y2]).reshape((1, -1))
        plot = np.concatenate([plot, plot_], axis=0)
        if index%2==0:
            cv2.putText(im_cv2,str(sign_num_h)+'-'+str(sign_num), (x1_ + 2, y1_ + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=5)

plot = plot[1::]
for area_index, area in enumerate(plot):
    x1, y1, x2, y2 = area
    w = x2 - x1
    h = y2 - y1
    roi = rawimg[y1:y1 + h, x1:x1 + w]
    plant_num = count_yield(roi)

    #plant_num=cv2.countNonZero(roi)
    x1_, y1_, x2_, y2_ = int(4*x1), int(4*y1), int(4*x2), int(4*y2)
    cv2.putText(im_cv2,str(plant_num), (x1_ + 2, y1_ + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=5)
cv2.imwrite('../kaiyuan/draw.png',im_cv2)