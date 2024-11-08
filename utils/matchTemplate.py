import cv2
import numpy as np
import time


to=time.time()

target = cv2.imread("/home/kingargroo/streamline/utils/te.jpg",0)
template  = cv2.imread("/home/kingargroo/streamline/utils/temp2.png",0)

# 創建SIFT檢測器
sift = cv2.SIFT_create()

# 檢測特徵點並計算描述子
kp1, des1 = sift.detectAndCompute(template, None)
kp2, des2 = sift.detectAndCompute(target, None)

# 使用暴力匹配器進行匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 應用比率測試
good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# 繪製匹配結果
if len(good) > 10:  # 如果有足夠的好匹配點，則繪製它們
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h,w = template.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    target = cv2.polylines(target, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0),  # 繪製匹配連接線的顏色
                       singlePointColor = None,
                       matchesMask = matchesMask,  # 繪製inlier匹配
                       flags = 2)

    result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    cv2.imwrite("target.jpg",result)
else:
    print("Not enough matches are found - {}/{}".format(len(good), 10))
    matchesMask = None
print(time.time()-to)