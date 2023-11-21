import onnxruntime as rt
import numpy as np
import cv2
from collections import Counter


def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # Intersection area
    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
            (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * np.square(np.arctan(w2 / h2) - np.arctan(w1 / h1))
                alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def find_most_common_elements(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]

# 前处理
def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    ih, iw, _ = image.shape
    print(ih, iw)
    h, w = size
    # letterbox_image = False
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # print(image.shape)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        image_back = image
        # cv2.imshow("img", image_back)
        # cv2.waitKey()
    return image_back


def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    return np.expand_dims(img, axis=0).astype(np.float32)


def std_output(pred):
    """
    将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred


def xywh2xyxy(*box):
    """
    将xywh转换为左上角点和左下角点
    Args:
        box:
    Returns: x1y1x2y2
    """
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
           box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret




def nms(pred, conf_thres, iou_thres):
    """
    非极大值抑制nms
    Args:
        pred: 模型输出特征图
        conf_thres: 置信度阈值
        iou_thres: iou阈值,列表，記錄對應class 的nms threshold ：{'corn':0.3,'cucumber':0.5,'wheat':0.7},即 [0.3,0.5,0.7]
    Returns: 输出后的结果
    """
    box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))

    pre_class=find_most_common_elements(cls)
    total_cls = list(set(cls))  # 记录图像内共出现几种物体
    output_box = []
    # 每个预测类别分开考虑
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            # 记录[x,y,w,h,conf(最大类别概率),class]值
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
        # box_conf_sort = np.argsort(-box_conf)
        # 得到置信度最大的预测框
        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)
        sort_cls_box = np.delete(sort_cls_box, 0, 0)
        # 对除max_conf_box外其他的框进行非极大值抑制
        while len(sort_cls_box) > 0:
            # 得到当前最大的框
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                #iou = get_iou(max_conf_box, current_box)
                iou=bbox_iou(max_conf_box,current_box,CIoU=True)
                if iou > iou_thres[pre_class]:
                    # 筛选出与当前最大框Iou大于阈值的框的索引
                    del_index.append(j)
            # 删除这些索引
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                # 我认为这里需要将clas_box先按置信度排序， 才能每次取第一个
                output_box.append(sort_cls_box[0])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box


def cod_trf(result, pre, after):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,
    """
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre / w_after, h_pre / h_after)  # 缩放比例
    h_pre, w_pre = h_pre / scale, w_pre / scale  # 计算原图在等比例缩放后的尺寸
    x_move, y_move = abs(w_pre - w_after) // 2, abs(h_pre - h_after) // 2  # 计算平移的量
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret


def draw(res, image, cls,show=False):
    """
    将预测框绘制在image上
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表，类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
    Returns:
    """
    for r in res:
        #draw point
        image=cv2.circle(image,center=(int((r[0]+r[2])/2),int((r[1]+r[3])/2)),radius=1,color=(0, 0, 255),thickness=10)
    text='number:{}'.format(len(res))
    imgh,imgw=image.shape[0:2]
    newh=imgh+350
    shape = (newh, imgw, 3)  # y, x, RGB
    # 直接建立全白圖片 100*100
    new_img = np.full(shape, 255)
    new_img[0:imgh,0:imgw,:]=image.copy()
    new_img=cv2.putText(new_img.astype(np.int32),text,(15,newh-230),cv2.FONT_HERSHEY_COMPLEX,5,(0, 0, 0), 10)
    new_img=cv2.putText(new_img.astype(np.int32),'id:{}'.format(1),(imgw-500,newh-230),cv2.FONT_HERSHEY_COMPLEX,5,(0, 0, 0), 10)
    if show:
        cv2.imshow("result", new_img)
        cv2.waitKey()
    return new_img,len(res)


def predict_single_img(input_path,img_path,onnx_model_path,out_path,class_list=['corn','sorghum','soybean','wheat']):
    std_h, std_w =1280,1280 # 标准输入尺寸

    class_list = class_list
    input_path = input_path
    img_path = img_path
    img = cv2.imread(input_path + img_path)
    if img.size == 0:
        print("error path")
    # 前处理
    img_after = resize_image(img, (std_w, std_h), True)  # （960， 1280， 3）
    # 将图像处理成输入的格式
    data = img2input(img_after)
    # 输入模型
    sess = rt.InferenceSession(onnx_model_path)  # yolov8模型onnx格式
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name: data})[0]  # 输出(8400x84, 84=80cls+4reg, 8400=3种尺度的特征图叠加), 这里的预测框的回归参数是xywh， 而不是中心点到框边界的距离
    pred = std_output(pred)
    # 置信度过滤+nms
    result = nms(pred, 0.15, [0.4,0.3,0.35,0.7])  # [x,y,w,h,conf(最大类别概率),class]
    # 坐标变换
    result = cod_trf(result, img, img_after)
    image,total_num = draw(result, img, class_list)
    # 保存输出图像
    out_path = out_path
    cv2.imwrite(out_path + img_path, image)
    return total_num






if __name__ == '__main__':
    total_num=predict_single_img(input_path="/home/kingargroo/seed/test/img/",img_path='test11.jpg',onnx_model_path="/home/kingargroo/seed/vision4.onnx",out_path="/home/kingargroo/seed/test/predict")
