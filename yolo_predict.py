import cv2
from ultralytics import YOLO
import os
from pathlib import Path

model=YOLO('/home/kingargroo/corn/runs/detect/train2/weights/best.pt')

img_path='/home/kingargroo/corn/M4'
img_list=os.listdir(img_path)

for img_name in img_list:

    img_name_=img_path+'/'+img_name
    img_title = img_name.split('.')[0]
    if img_name.split('.')[1]!='png':
        break
    img=cv2.imread(img_name_)


    #print(img_title)

    txt_name='/home/kingargroo/corn/new_result/'+str(img_title)+'.txt'
    results = model(img, stream=True)
    txt_file=open(txt_name,'a+')

    for result in results:
        boxes = result.boxes
        for box in boxes:
            score=box.conf[0].cpu().numpy()
            xc, yc, w, h = box.xywhn[0]
            xc, yc, w, h =xc.cpu().numpy(), yc.cpu().numpy(), w.cpu().numpy(), h.cpu().numpy()
            output_str='0'+' '+str(xc)+' '+str(yc)+' '+str(w)+' '+str(h)+' '+str(score)+'\n'
            txt_file.write(output_str)





