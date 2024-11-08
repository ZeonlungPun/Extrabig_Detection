import cv2
from ultralytics import YOLO
import os,time,torch
import numpy as np

def non_max_suppression(boxes, probs=[],overlapThresh=0.8):

    len_init = len(boxes)
    print("NMS is doing, initial bounding box number：{}".format(len_init))
    #如果输入的坐标组没有对象，直接返回空列表
    if len(boxes) == 0:
        return [], [], []

    #boxs转array
    boxes = np.asarray([b[:4] for b in boxes])
    #将boxes里的数据类型转换为float类型
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    #初始化所选索引的列表
    pick = []
    #获取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #计算框面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    #如果置信度为空，根据边界框的右下角y坐标对框进行排序
    if len(probs) == 0:
        idxs = np.argsort(y2)
    #否则，按最高prob（降序）对框进行排序
    else:
        # idxs = np.argsort(probs)[::-1]
        idxs = np.argsort(probs)

    #保持循环，同时某些索引仍保留在索引中
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]

        pick.append(i)

        #将拿到的对象和其他目标做坐标方面的对比，x1[i]比x2[i]小，y1[i]比y2[i]小
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

            #删掉
        idxs = np.delete(
                idxs,
                np.concatenate(([last], np.where(overlap > overlapThresh)[0])))


    print("after NMS,bounding box number is :", len(pick))
    return pick

def predict_with_yolov8(model_name,img_path,save_path,with_score=True):
    print(torch.cuda.is_available())
    t0 = time.time()
    img_list=os.listdir(img_path)
    print('the amount of images needed to be predict is:',len(img_list))
    for img_name in img_list:
        img_name_=img_path+'/'+img_name
        img_title = img_name.split('.')[0]
        img=cv2.imread(img_name_)

        txt_name=save_path+str(img_title)+'.txt'
        txt_file = open(txt_name, 'a+')

        model=YOLO(model_name)
        results = model(img, stream=True)

        boxes_before_nms = []
        normalize_before_nms = []
        scores = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()
                boxes_before_nms.append(np.array([x1, y1, x2, y2]))
                x1, y1, x2, y2 = box.xyxyn[0]
                x1, y1, x2, y2 = x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()
                normalize_before_nms.append(np.array([x1, y1, x2, y2]))
                score=box.conf[0].cpu().numpy()
                scores.append(score)
                #xc, yc, w, h = box.xywhn[0]
                #xc, yc, w, h =xc.cpu().numpy(), yc.cpu().numpy(), w.cpu().numpy(), h.cpu().numpy()
        #NMS
        boxes_before_nms = np.array(boxes_before_nms)
        pick_idx = non_max_suppression(boxes_before_nms, probs=scores, overlapThresh=0.6)
        #boxes_before_nms,scores=np.array(boxes_before_nms),np.array(scores)
        #pick_idx  =cv2.dnn.NMSBoxes(bboxes=boxes_before_nms,scores=scores,score_threshold=0,nms_threshold=0.6)
        normalize_before_nms = np.array(normalize_before_nms)
        scores=np.array(scores)
        pick_idx = np.array(pick_idx).astype(int)
        boxes_after_nms = normalize_before_nms[pick_idx]
        scores_after_nms=scores[pick_idx]
        try:
            for box,score in zip(boxes_after_nms,scores_after_nms):
                x1, y1, x2, y2 = box
                xc, yc, w, h = (x1 + x2) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1

                if with_score:
                    output_str='0'+' '+str(xc)+' '+str(yc)+' '+str(w)+' '+str(h)+' '+str(score)+'\n'
                else:
                    output_str = '0' + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) +  '\n'
                txt_file.write(output_str)
        except:
            continue

    t1=time.time()
    print('total prediction time is :',t1-t0)

if __name__ == '__main__':
    model = '/home/kingargroo/corn/runs/detect/train19/weights/best.pt'
    img_path = '/home/kingargroo/corn/xinmi3/img'
    save_path = '/home/kingargroo/corn/xinmi3/result/'
    predict_with_yolov8(model,img_path,save_path,with_score=True)
