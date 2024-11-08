import numpy as np
import cv2
import pandas as pd

def Visualize(save_path,vertical_lines,horizontal_lines,img):

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

        sign_num_h=horizontal_lines.shape[0]
        for j in range(horizontal_lines.shape[0]-1):

            sign_num_h-=1
            y1 = horizontal_lines[j]
            y2 = horizontal_lines[j + 1]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, thickness=1)
            plot_ = np.array([x1, y1, x2, y2]).reshape((1, -1))
            plot = np.concatenate([plot, plot_], axis=0)
            # if index%2==0:
            #     cv2.putText(img,str(sign_num_h)+'-'+str(sign_num), (x1 + 2, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255, thickness=1)
            if i%2==0:
                coor_text=str(j)+str('.')+str(i)
                cv2.putText(img,coor_text,(x1 + 2, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,255, thickness=1)

    plot = plot[1::]
    print(plot)
    cv2.imwrite(save_path, img)
    return plot
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


def Count(save_path,plot,raw_img):
    # count the plant number in each plot
    plot_number = np.zeros((plot.shape[0], 5))
    for area_index, area in enumerate(plot):
        x1, y1, x2, y2 = area
        w = x2 - x1
        h = y2 - y1
        roi = raw_img[y1:y1 + h, x1:x1 + w]
        plant_num = count_yield(roi)
        #plant_num = cv2.countNonZero(roi)/255
        plot_number[area_index, 4] = plant_num
        plot_number[area_index, 0:4] = np.array([x1, y1, x2, y2]).reshape((1, -1))

    df = pd.DataFrame(plot_number)
    df.columns = ['x1', 'y1', 'x2', 'y2', 'count']
    df.to_csv(save_path)
    return df

def SegemtPlot(img):
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h,w=img.shape[0:2]

    horizontal_lines=[]
    h_=1
    # add first horizontal lines
    horizontal_lines.append(h_)

    while True:
        h_+=100
        if h_+100>h:
            break
        # find the minimun value in 100 pixel range
        pixel_list,index_list=[],[]
        for i in range(150):
            total_pixel=np.sum(img[h_+i:h_+i+3,:])
            pixel_list.append(total_pixel)
            index_list.append(h_+i)
        min_index=np.argmin(pixel_list)
        min_h=index_list[min_index]
        if min_h-horizontal_lines[-1] <200:
            continue
        horizontal_lines.append(min_h+10)
    # add last horizontal lines
    horizontal_lines.append(h-5)
    print('h:',horizontal_lines)

    w_=1
    vertical_lines=[]
    vertical_lines.append(w_)

    while True:
        w_+=10
        if w_+10>w:
            break
        pixel_list, index_list = [], []
        for i in range(10):
            total_pixel=np.sum(img[:,w_+i:w_+i+3])
            pixel_list.append(total_pixel)
            index_list.append(w_ + i)
        min_index = np.argmin(pixel_list)
        min_w = index_list[min_index]
        if min_w-vertical_lines[-1]<10:
            continue
        vertical_lines.append(min_w+2)
    vertical_lines.append(w-2)
    print('v:',vertical_lines)
    return horizontal_lines,vertical_lines


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
def SegmentMain(rawimg,Segment_plot_path,First_count_save_path):
    """
    :param rawimg:  the calibrated image after get the ROI,only 1 channel
    :param Segment_plot_path: the result image saving path of segmentation
    :param First_count_save_path:
    :return: initial counting result
    """
    assert len(rawimg.shape)==2
    horizontal_lines, vertical_lines = SegemtPlot(rawimg)
    vertical_lines = check_plot(vertical_lines, rawimg)
    #we two raw image here , one for count and one for visualization
    count_img=rawimg.copy()
    plot = Visualize(Segment_plot_path, vertical_lines, horizontal_lines, rawimg)
    plot_number=Count(save_path=First_count_save_path, plot=plot, raw_img=count_img)
    return plot_number,horizontal_lines, vertical_lines

def has_three_consecutive_zeros(lst):
    for i in range(len(lst) - 3):
        if lst[i] == 0 and lst[i+1] == 0 and lst[i+2] == 0 and lst[i+3] == 0:
            return True
    return False

def Check_Consecutive(vis_img,img,plot_info,vis_img_save_path):
    flag=[]
    for data in plot_info.iterrows():
        data=data[1]
        x1, y1, x2, y2 =int(data.iloc[1]),int(data.iloc[2]),int(data.iloc[3]),int(data.iloc[4])
        y_list=list(np.linspace(start=y1,stop=y2,num=22))
        continue_zero_plot = []
        for y_ in y_list[1::]:
            y_pred=y_list[y_list.index(y_)-1]
            y_,y_pred=int(y_),int(y_pred)
            cv2.line(vis_img,pt1=(x1,y_),pt2=(x2,y_),color=255,thickness=1)
            roi = img[y_pred:y_, x1:x2]

            plant_num = cv2.countNonZero(roi)
            continue_zero_plot.append(plant_num)
        if has_three_consecutive_zeros(continue_zero_plot):
            print("warning plot:",data)
            flag.append(1)
        else:
            flag.append(0)
    cv2.imwrite(vis_img_save_path,vis_img)
    flag=np.array(flag).reshape((-1,1))
    flag=pd.DataFrame(flag,columns=["flag"])
    plot_info=pd.concat([plot_info,flag],axis=1)
    return plot_info


if __name__ =='__main__':
    img_path='/home/kingargroo/streamline/LINHAI1/linghaizhengqu15-1_point.png'
    raw_img=cv2.imread(img_path,0)
    #raw_img=cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
    print(raw_img.shape)
    Segment_plot_path='slide2.jpg'
    Fist_count_save_path='countresults1.csv'
    plot_number,horizontal_lines, vertical_lines=SegmentMain(rawimg=raw_img,Segment_plot_path=Segment_plot_path,First_count_save_path=Fist_count_save_path)


