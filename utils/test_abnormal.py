import cv2
import numpy as np
import pandas as pd

# def has_three_consecutive_zeros(lst):
#     for i in range(len(lst) - 2):
#         if lst[i] == 0 and lst[i+1] == 0 and lst[i+2] == 0:
#             return True
#     return False
#
# plot_info=pd.read_csv('/home/kingargroo/streamline/folder2/countresult1.csv')
# img=cv2.imread('/home/kingargroo/streamline/folder2/transform.jpg',0)
# vis_img=cv2.imread('/home/kingargroo/streamline/folder2/plot_result.jpg',0)
# flag=[]
# for data in plot_info.iterrows():
#     data=data[1]
#     x1, y1, x2, y2 =int(data.iloc[1]),int(data.iloc[2]),int(data.iloc[3]),int(data.iloc[4])
#     y_list=list(np.linspace(start=y1,stop=y2,num=22))
#     continue_zero_plot = []
#     for y_ in y_list[1::]:
#         y_pred=y_list[y_list.index(y_)-1]
#         y_,y_pred=int(y_),int(y_pred)
#         cv2.line(vis_img,pt1=(x1,y_),pt2=(x2,y_),color=255,thickness=1)
#         roi = img[y_pred:y_, x1:x2]
#
#         plant_num = cv2.countNonZero(roi)
#         continue_zero_plot.append(plant_num)
#     if has_three_consecutive_zeros(continue_zero_plot):
#         print("warning plot:",data)
#         flag.append(1)
#     else:
#         flag.append(0)
#
# flag=np.array(flag).reshape((-1,1))
# flag=pd.DataFrame(flag,columns=["flag"])
# plot_info=pd.concat([plot_info,flag],axis=1)

