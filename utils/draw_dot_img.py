import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw,  ImageFont
import numpy as np
from skimage import io
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

def plot_detections(im, boxes,
                    scores=[],
                    classes=[],
                    outfile='',
                    plot_thresh=0.2,
                    color_dict={},
                    plot_line_thickness=1, show_labels=True,
                    compression_level=9,
                    skip_empty=False,
                    test_box_rescale_frac=1,
                    label_txt=None,
                    draw_rect=True, draw_circle=False,reduce=True):



    #预测框、字体等参数的设置
    font_size = 1.5
    font_width = 2
    display_str_height = 9
    font = cv2.FONT_HERSHEY_SIMPLEX

    output = im
    nboxes = 0


    alpha = 1
    for box, score, classy in zip(boxes, scores, classes):

        if score >= plot_thresh:
            nboxes += 1
            [xmin, ymin, xmax, ymax] = box
            left, right, top, bottom = xmin, xmax, ymin, ymax

            # 获得标签和对应的颜色
            classy_str = str(classy) + ': ' + \
                         str(int(100 * float(score))) + '%'
            color = color_dict[classy]


            #绘制矩形框
            if draw_rect:
                if reduce==True:
                    cv2.rectangle(output, (int(left/4), int(bottom/4)), (int(right/4),
                                                                     int(top/4)), color,
                                  plot_line_thickness)


                else:

                    cv2.rectangle(output, (int(left), int(bottom)), (int(right),
                                                                 int(top)), color,
                              plot_line_thickness)

            #绘制圆形框
            if draw_circle:  # 改一下，半径指定为定值，即下面的r
                d = max(abs(left - right), abs(top - bottom))  # linetype是调整线条粗细的，负值为实心
                # r = int(d/2.0)
                r = 2
                cx, cy = int((left + right) / 2.0), int((top + bottom) / 2.0)
                cv2.circle(output, (cx, cy), r, color, plot_line_thickness, lineType=-1)

            #是否显示标签(即预测框上面的类别)
            if show_labels:
                # 获取位置
                display_str = classy_str
                total_display_str_height = (1 + 2 * 0.05) * display_str_height
                if top > total_display_str_height:
                    text_bottom = top
                else:
                    text_bottom = bottom + total_display_str_height
                #反转列表并从下至上打印。
                (text_width, text_height), _ = cv2.getTextSize(display_str,
                                                               fontFace=font,
                                                               fontScale=font_size,
                                                               thickness=font_width)
                margin = np.ceil(0.1 * text_height)

                #获取矩形坐标和文本坐标，
                rect_top_left = (int(left - (plot_line_thickness - 1) * margin),
                                 int(text_bottom - text_height - (plot_line_thickness + 3) * margin))
                rect_bottom_right = (int(left + text_width + margin),
                                     int(text_bottom - (plot_line_thickness * margin)))
                text_loc = (int(left + margin),
                            int(text_bottom - (plot_line_thickness + 2) * margin))

                #需要显示标签时，以下语句可以让标签稍微下移一些
                if (alpha > 0.75) and (plot_line_thickness > 1):
                    rect_top_left = (rect_top_left[0], int(
                        rect_top_left[1] + margin))
                    rect_bottom_right = (rect_bottom_right[0], int(
                        rect_bottom_right[1] + margin))
                    text_loc = (text_loc[0], int(text_loc[1] + margin))

                if draw_rect:
                    cv2.rectangle(output, rect_top_left, rect_bottom_right,
                                  color, -1)
                cv2.putText(output, display_str, text_loc,
                            font, font_size, (0, 0, 0), font_width, cv2.LINE_AA)


    #调整输出图像大小
    if test_box_rescale_frac != 1:
        height, width = output.shape[:2]
        output = cv2.resize(output, (width / test_box_rescale_frac, height / test_box_rescale_frac),
                            interpolation=cv2.INTER_CUBIC)

    #如果需要，给图像添加txt文本，在text_loc_label位置写
    if label_txt:
        text_loc_label = (10, 20)
        cv2.putText(output, label_txt, text_loc_label,
                    font, 2 * font_size, (0, 0, 0), font_width,
                    cv2.LINE_AA)

    #保存压缩后的图片，compression_level为缩放系数
    if skip_empty and nboxes == 0:
        return
    else:
        print("结果图绘制完成，保存在:", outfile)
        cv2.imwrite(outfile, output, [
            cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return

# df_refine=pd.read_csv('/home/kingargroo/corn/out_results/M44/params.csv')
# original = '/home/kingargroo/corn/big_img'
# out_results='/home/kingargroo/corn/out_results/M44'
# plot_dir='M44'
def DrawAuxiliaryImages(df_refine_path,original_path,out_results_path,detect_thresh=0.2,im_ext='.png',reduce=True):
    """
    :param df_refine_path:   a csv file that store some key parameters for detection
    :param original_path:  original big image path
    :param out_results_path: save the drawed plot to this path (save the final result)
    :param detect_thresh:  lower this detection confidence score will be neglected
    :param im_ext: big image format
    :param reduce:  Four-dimensional downsampling to read in the image or not
    """
    df_refine=pd.read_csv(df_refine_path)
    colors = 40*[(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 140, 255),
                  (0, 255, 125), (125, 125, 125)]
    color_dict = {}
    for i, c in enumerate(sorted(np.unique(df_refine['category'].values))):
        color_dict[c] = colors[i]
    original=original_path
    plot_path = out_results_path
    # 获得输入图片的路径、根据标签文件生成的路径
    im_names_tiled = sorted([z.split(im_ext)[0] for z in os.listdir(original) if z.endswith(im_ext)])
    im_names_set = set(df_refine['im_name_root'].values)
    print(im_names_set)
    print(im_names_tiled)
    if len(im_names_set) == 0:
        print("路径中不存在标签对应的图片", original, "其后缀为:", im_ext)
        print("返回中")


    names = ['X', 'Y']
    for i, im_name in enumerate(im_names_tiled):

        im_path = os.path.join(original, im_name + im_ext)
        outfile_plot_image = os.path.join(plot_path, im_name + '.jpg')
        outfile_point = os.path.join(plot_path, im_name + '_coordinate' + '.csv')
        outfile_point_img = os.path.join(plot_path, im_name + '_point' + '.png')

        print("绘制图像:", im_name)

        if reduce == True:
            im_cv2 = cv2.imread(im_path, cv2.IMREAD_REDUCED_COLOR_4)
        else:
            #im_cv2 = cv2.imread(im_path)
            im_pil = Image.open(im_path)
            # 将PIL图像转换为OpenCV格式（如果需要的话）
            im_cv2 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

        x_pos = []
        y_pos = []
        X_Y = []

        if im_name not in im_names_set:
            boxes, probs, classes, box_names = [], [], [], []
        else:
            df_filt = df_refine[df_refine['im_name_root'] == im_name]
            boxes = df_filt[['Xmin_Glob', 'Ymin_Glob', 'Xmax_Glob', 'Ymax_Glob']].values
            probs = df_filt['prob']
            classes = df_filt['category']
            if reduce == True:
            # 黑底图，后续用于显示植株点，四倍下採樣
                img_black=cv2.imread(im_path, cv2.IMREAD_REDUCED_COLOR_4)
            else:
                img_black = cv2.imread(im_path)

            point_img = np.zeros((img_black.shape[0], img_black.shape[1]), dtype=np.uint8)
            for box in boxes:
                [xmin, ymin, xmax, ymax] = box
                left, right, top, bottom = xmin, xmax, ymin, ymax
                # 计算中心点坐标
                cx, cy = int((left + right) / 2.0), int((top + bottom) / 2.0)

                if reduce==True:
                    cx, cy = int(cx / 4), int(cy / 4)
                point_img[cy, cx] = 255
                x_pos.append(cx)
                y_pos.append(cy)
                X_Y.append((cx, cy))

            out_paint = pd.DataFrame(columns=names, data=X_Y)
            out_paint.to_csv(outfile_point)
            print(point_img.shape)
            cv2.imwrite(outfile_point_img, point_img)

            plot_detections(im_cv2, boxes,
                            scores=probs,
                            classes=classes,
                            outfile=outfile_plot_image,
                            plot_thresh=detect_thresh,
                            color_dict=color_dict,
                            plot_line_thickness=1,
                            show_labels=False,
                            test_box_rescale_frac=1,
                            label_txt=None,
                            draw_rect=True, draw_circle=False,reduce=reduce)



