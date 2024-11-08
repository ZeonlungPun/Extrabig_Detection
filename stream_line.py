"""
this file deal with the whole process of the maize seedlings yield prediction,
including slice the big image, detection , generate plot image ,calibration ,extract the plot ,give the exact nuber of yield
"""
import sys,os
import cv2,time
sys.path.append('.')
from utils import *
import numpy as np
import pandas as pd
import argparse
parser=argparse.ArgumentParser(description='parameters of maize seedlings yield prediction project')
"""
add the parameters model needed
mainly the saving path for all the processes 
"""

import argparse
parser=argparse.ArgumentParser(description='parameters of maize seedlings yield prediction project')


parser.add_argument('--target_path',type=str,default='/home/kingargroo/arieal/big', help='local big image path')
parser.add_argument('--out_folder',type=str,default='/home/kingargroo/arieal/img_patch', help='the save path of image patches')
parser.add_argument('--in_ext',type=str, default='tif', help='the suffix of the big image')
parser.add_argument('--local_pt_path',type=str, default='/home/kingargroo/arieal/best.pt', help='the path of local model weight')
parser.add_argument('--classes_list',type=list,default=['M'],help='class list of detection model')
parser.add_argument('--save_path',type=str,default='/home/kingargroo/arieal/predict/',help='where the prediction txt path save')
parser.add_argument('--out_path',type=str,default='/home/kingargroo/arieal/big/stream', help='result saving path,including some intermediate results')
parser.add_argument('--farmland_information_file',type=str,default='data.csv',help='the CSV file of farmland information, needed to extract four_lines number')
parser.add_argument('--place',type=str,default='xinmi')

# parse the parameters
args = parser.parse_args(args=[])

# # # 调用裁图函数
#slice_to_png(tif_path=args.target_path, out_path=args.out_folder, in_ext=args.in_ext)


# # # # 调用yolov8预测函数
#predict_with_yolov8(args.local_pt_path, args.out_folder, args.save_path, with_score=True)

# # # # 调用后处理函数
#execute(labels_dir=args.save_path, original=args.target_path, out_results=args.out_path, classes=args.classes_list, im_ext='.tif')

# # # 调用绘图函数，生成可视结果图、csv文件、
# df_refine_path = args.out_path + '/' + 'params.csv'
# DrawAuxiliaryImages(df_refine_path=df_refine_path, original_path=args.target_path,
#                         out_results_path=args.out_path, reduce=True,
#                        detect_thresh=0.25,im_ext='.tif')
dot_img_path = args.out_path+ '/' + args.place+'_point.png'
print(dot_img_path)
dot_img = cv2.imread(dot_img_path, cv2.IMREAD_GRAYSCALE)
# #讀取端點數據
place=args.target_path.split('/')[-2].split('_')[-1]
point_path=os.path.join(args.target_path,args.place+'_'+'points.csv')
point_df=pd.read_csv(point_path,header=None)
#4倍下採樣，爲黑圖的仿射變換作準備
point_array=np.array(point_df/4,dtype=int)


trans = ImageTransform(src_img=dot_img, input_points=point_array, transform_category='Perspective',
                       save=True, save_path=args.out_path)
roi = trans.main()

#畫分好小區的可視化圖保存位置
Segment_plot_path=args.out_path + '/'+"plot_result.jpg"
#中間計數文件
First_count_save_path=args.out_path + '/'+"countresult1.csv"

# 分区并计算出苗数
plot_number,horizontal_lines, vertical_lines = SegmentMain(rawimg=roi, Segment_plot_path=Segment_plot_path,
                              First_count_save_path=First_count_save_path)

#檢查廢區
vis_img=cv2.imread(Segment_plot_path,0)
img=cv2.imread(dot_img_path,0)
plot_info=pd.read_csv(args.out_path + '/'+'countresult1.csv')
vis_img_save_path=args.out_path+ '/' +'vis_img.jpg'
plot_info=Check_Consecutive(vis_img,img,plot_info,vis_img_save_path)
plot_info.to_csv(args.out_path + '/'+'countresult1.csv')
# 根据田图信息，生成出苗数文本数据
#final_output,two_line_count,four_line_count=GetStandardOutput(csv_name=First_count_save_path,location=args.target_place,location_file=args.farmland_information_file)
#final_output.to_csv(os.path.join(args.out_path,'final_output.csv'))



