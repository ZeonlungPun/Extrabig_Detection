import cv2


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('point is:',(x,y))

def mouse(event, x, y, flags, param):
    global flag, x1, y1, x2, y2, wx, wy, move_w, move_h, dst
    global zoom, zoom_w, zoom_h, img_zoom, flag_har, flag_var
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        if flag == 0:
            flag = 1
            x1, y1, x2, y2 = x, y, wx, wy  # 使鼠标移动距离都是相对于初始点击位置，而不是相对于上一位置
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        if flag == 1:
            move_w, move_h = x1 - x, y1 - y  # 鼠标拖拽移动的宽高
            if flag_har and flag_var:  # 当窗口宽高大于图片宽高
                wx = x2 + move_w  # 窗口在大图的横坐标
                if wx < 0:  # 矫正位置
                    wx = 0
                elif wx + win_w > zoom_w:
                    wx = zoom_w - win_w
                wy = y2 + move_h  # 窗口在大图的总坐标
                if wy < 0:
                    wy = 0
                elif wy + win_h > zoom_h:
                    wy = zoom_h - win_h
                dst = img_zoom[wy:wy + win_h, wx:wx + win_w]  # 截取窗口显示区域
            elif flag_har and flag_var == 0:  # 当窗口宽度大于图片宽度
                wx = x2 + move_w
                if wx < 0:
                    wx = 0
                elif wx + win_w > zoom_w:
                    wx = zoom_w - win_w
                dst = img_zoom[0:zoom_h, wx:wx + win_w]
            elif flag_har == 0 and flag_var:  # 当窗口高度大于图片高度
                wy = y2 + move_h
                if wy < 0:
                    wy = 0
                elif wy + win_h > zoom_h:
                    wy = zoom_h - win_h
                dst = img_zoom[wy:wy + win_h, 0:zoom_w]
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        flag = 0
        x1, y1, x2, y2 = 0, 0, 0, 0
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
        z = zoom
        if flags > 0:  # 滚轮上移
            zoom += wheel_step
            if zoom > 1 + wheel_step * 20:  # 缩放倍数调整
                zoom = 1 + wheel_step * 20
        else:  # 滚轮下移
            zoom -= wheel_step
            if zoom < wheel_step:  # 缩放倍数调整
                zoom = wheel_step
        zoom = round(zoom, 2)  # 取2位有效数字
        zoom_w, zoom_h = int(img_original_w * zoom), int(img_original_h * zoom)
        # print(wx, wy)
        wx, wy = int((wx + x) * zoom / z - x), int((wy + y) * zoom / z - y)  # 缩放后窗口在图片中的坐标
        # print(z, zoom, x, y, wx, wy)
        if wx < 0:
            wx = 0
        elif wx + win_w > zoom_w:
            wx = zoom_w - win_w
        if wy < 0:
            wy = 0
        elif wy + win_h > zoom_h:
            wy = zoom_h - win_h
        img_zoom = cv2.resize(img_original, (zoom_w, zoom_h), interpolation=cv2.INTER_AREA)  # 图片缩放
        if zoom_w <= win_w and zoom_h <= win_h:  # 缩放后图片宽高小于窗口宽高
            flag_har, flag_var = 0, 0
            dst = img_zoom
            cv2.resizeWindow('img', zoom_w, zoom_h)
        elif zoom_w <= win_w and zoom_h > win_h:  # 缩放后图片宽度小于窗口宽度
            flag_har, flag_var = 0, 1
            dst = img_zoom[wy:wy + win_h, 0:zoom_w]
            cv2.resizeWindow('img', zoom_w, win_h)
        elif zoom_w > win_w and zoom_h <= win_h:  # 缩放后图片高度小于窗口高度
            flag_har, flag_var = 1, 0
            dst = img_zoom[0:zoom_h, wx:wx + win_w]
            cv2.resizeWindow('img', win_w, zoom_h)
        else:  # 缩放后图片宽高大于于窗口宽高
            flag_har, flag_var = 1, 1
            dst = img_zoom[wy:wy + win_h, wx:wx + win_w]
            cv2.resizeWindow('img', win_w, win_h)
    cv2.imshow("img", dst)
    cv2.waitKey(1)


# win_h, win_w = 11297, 24583  # 窗口宽高
# wx, wy = 0, 0  # 窗口相对于原图的坐标
# wheel_step, zoom = 0.05, 1  # 缩放系数， 缩放值
# img_original = cv2.imread("/home/kingargroo/arieal/xinmi.tif")  # 建议图片大于win_w*win_h(800*600)
# img_original_h, img_original_w = img_original.shape[0:2]  # 原图宽高
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# cv2.moveWindow("img", 300, 100)
# zoom_w, zoom_h = img_original_w, img_original_h  # 缩放图宽高
# img_zoom = img_original.copy()  # 缩放图片
# flag, flag_har, flag_var = 0, 0, 0  # 鼠标操作类型
# move_w, move_h = 0, 0  # 鼠标移动坐标
# x1, y1, x2, y2 = 0, 0, 0, 0  # 中间变量
# cv2.resizeWindow("img", win_w, win_h)
# dst = img_original[wy:wy + win_h, wx:wx + win_w]
# cv2.setMouseCallback('img', mouse)
# if img_original_w > win_w:
#     flag_har = 1
# if img_original_h > win_h:
#     flag_var = 1
# cv2.waitKey()
# cv2.destroyAllWindows()




##############################################################################




import cv2

# 全局变量
g_window_name = "img"  # 窗口名
g_window_wh = [24583, 11297]  # 窗口宽高

g_location_win = [0, 0]  # 相对于大图，窗口在图片中的位置
location_win = [0, 0]  # 鼠标左键点击时，暂存g_location_win
g_location_click, g_location_release = [0, 0], [0, 0]  # 相对于窗口，鼠标左键点击和释放的位置

g_zoom, g_step = 1, 0.1  # 图片缩放比例和缩放系数
g_image_original = cv2.imread("/home/kingargroo/arieal/xinmi.tif")  # 原始图片，建议大于窗口宽高（800*600）
g_image_zoom = g_image_original.copy()  # 缩放后的图片
g_image_show = g_image_original[g_location_win[1]:g_location_win[1] + g_window_wh[1],
               g_location_win[0]:g_location_win[0] + g_window_wh[0]]  # 实际显示的图片


# 矫正窗口在图片中的位置
# img_wh:图片的宽高, win_wh:窗口的宽高, win_xy:窗口在图片的位置
def check_location(img_wh, win_wh, win_xy):
    for i in range(2):
        if win_xy[i] < 0:
            win_xy[i] = 0
        elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
            win_xy[i] = img_wh[i] - win_wh[i]
        elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
            win_xy[i] = 0
    # print(img_wh, win_wh, win_xy)


# 计算缩放倍数
# flag：鼠标滚轮上移或下移的标识, step：缩放系数，滚轮每步缩放0.1, zoom：缩放倍数
def count_zoom(flag, step, zoom):
    if flag > 0:  # 滚轮上移
        zoom += step
        if zoom > 1 + step * 20:  # 最多只能放大到3倍
            zoom = 1 + step * 20
    else:  # 滚轮下移
        zoom -= step
        if zoom < step:  # 最多只能缩小到0.1倍
            zoom = step
    zoom = round(zoom, 2)  # 取2位有效数字
    return zoom


# OpenCV鼠标事件
def mouse(event, x, y, flags, param):
    global g_location_click, g_location_release, g_image_show, g_image_zoom, g_location_win, location_win, g_zoom
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        g_location_click = [x, y]  # 左键点击时，鼠标相对于窗口的坐标
        location_win = [g_location_win[0], g_location_win[1]]  # 窗口相对于图片的坐标，不能写成location_win = g_location_win
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        g_location_release = [x, y]  # 左键拖曳时，鼠标相对于窗口的坐标
        h1, w1 = g_image_zoom.shape[0:2]  # 缩放图片的宽高
        w2, h2 = g_window_wh  # 窗口的宽高
        show_wh = [0, 0]  # 实际显示图片的宽高
        if w1 < w2 and h1 < h2:  # 图片的宽高小于窗口宽高，无法移动
            show_wh = [w1, h1]
            g_location_win = [0, 0]
        elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
            show_wh = [w2, h1]
            g_location_win[0] = location_win[0] + g_location_click[0] - g_location_release[0]
        elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
            show_wh = [w1, h2]
            g_location_win[1] = location_win[1] + g_location_click[1] - g_location_release[1]
        else:  # 图片的宽高大于窗口宽高，可左右上下移动
            show_wh = [w2, h2]
            g_location_win[0] = location_win[0] + g_location_click[0] - g_location_release[0]
            g_location_win[1] = location_win[1] + g_location_click[1] - g_location_release[1]
        check_location([w1, h1], [w2, h2], g_location_win)  # 矫正窗口在图片中的位置
        g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] + show_wh[1],
                       g_location_win[0]:g_location_win[0] + show_wh[0]]  # 实际显示的图片
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
        z = g_zoom  # 缩放前的缩放倍数，用于计算缩放后窗口在图片中的位置
        g_zoom = count_zoom(flags, g_step, g_zoom)  # 计算缩放倍数
        w1, h1 = [int(g_image_original.shape[1] * g_zoom), int(g_image_original.shape[0] * g_zoom)]  # 缩放图片的宽高
        w2, h2 = g_window_wh  # 窗口的宽高
        g_image_zoom = cv2.resize(g_image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
        show_wh = [0, 0]  # 实际显示图片的宽高
        if w1 < w2 and h1 < h2:  # 缩放后，图片宽高小于窗口宽高
            show_wh = [w1, h1]
            cv2.resizeWindow(g_window_name, w1, h1)
        elif w1 >= w2 and h1 < h2:  # 缩放后，图片高度小于窗口高度
            show_wh = [w2, h1]
            cv2.resizeWindow(g_window_name, w2, h1)
        elif w1 < w2 and h1 >= h2:  # 缩放后，图片宽度小于窗口宽度
            show_wh = [w1, h2]
            cv2.resizeWindow(g_window_name, w1, h2)
        else:  # 缩放后，图片宽高大于窗口宽高
            show_wh = [w2, h2]
            cv2.resizeWindow(g_window_name, w2, h2)
        g_location_win = [int((g_location_win[0] + x) * g_zoom / z - x),
                          int((g_location_win[1] + y) * g_zoom / z - y)]  # 缩放后，窗口在图片的位置
        check_location([w1, h1], [w2, h2], g_location_win)  # 矫正窗口在图片中的位置
        # print(g_location_win, show_wh)
        g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] + show_wh[1],
                       g_location_win[0]:g_location_win[0] + show_wh[0]]  # 实际的显示图片
    cv2.imshow(g_window_name, g_image_show)


# 主函数
if __name__ == "__main__":
    # 设置窗口
    cv2.namedWindow(g_window_name, cv2.WINDOW_NORMAL)
    # 设置窗口大小，只有当图片大于窗口时才能移动图片
    cv2.resizeWindow(g_window_name, g_window_wh[0], g_window_wh[1])
    cv2.moveWindow(g_window_name, 700, 100)  # 设置窗口在电脑屏幕中的位置
    # 鼠标事件的回调函数
    cv2.setMouseCallback(g_window_name, mouse)
    cv2.waitKey()  # 不可缺少，用于刷新图片，等待鼠标操作
    cv2.destroyAllWindows()