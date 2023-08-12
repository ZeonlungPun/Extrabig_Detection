import cv2,time
import numpy as np


raw_img=cv2.imread('circle.jpg')
copy_img=raw_img.copy()
copy_img2=raw_img.copy()

t1=time.time()
#prepose the image
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

ret, raw_img = cv2.threshold(raw_img, 100, 255, 0)
raw_img = cv2.medianBlur(raw_img, 5)

kernel = np.ones((5, 5), dtype=np.uint8)
raw_img = cv2.erode(raw_img, kernel, iterations=5)

edges=cv2.Canny(raw_img,100,200,apertureSize=3)
#150 150 80
#200 150 80
lines = cv2.HoughLinesP(edges, 10, np.pi / 180, 200, np.array([]), minLineLength=1, maxLineGap=1890)
#print(lines)
vertical_lines=[]
gap_lin_x=np.zeros((1,1))

for line in lines:
    x1, y1, x2, y2 = line[0]
    #calculate slope
    k=(y2-y1)/(x2-x1+0.001)
    #convert to arch
    angle=np.arctan(k)*180 / np.pi

    length = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

    #get vertical line 90°/-90°/270°/-270°
    if  (angle >=89 and  angle <=91 ) or (angle >= -91 and  angle <= -89 ) or (angle >=-271 and  angle <=-269 ) or (angle >=269 and  angle <=271 ):

        if length> 8000:
            x_center=(x1+x2)/2

            #only keep one center line in each seperated region
            gap=abs(x_center-gap_lin_x)
            gap_judge=(gap<50)
            if np.sum(gap_judge) >0:
                continue
            else:
                #keep the seperated line
                gap_lin_x=np.concatenate([gap_lin_x,x_center.reshape((1,1))],axis=0)
                #draw the between line
                cv2.line(copy_img, (x1+25, y1), (x2+25, y2), (0, 0, 255), thickness=1)
                vertical_lines.append([x1+25,y1,x2+25,y2,length])


#horizontal seperation
gap_lin_y=np.zeros((1,1))

h,w =raw_img.shape[0:2]
horizontal_lines=[]
#top and bottom
cv2.line(copy_img,(1,10),(w,10),(0,0,255),thickness=1)
cv2.line(copy_img,(1,h-10),(w,h-10),(0,0,255),thickness=1)
horizontal_lines.append([1,10,w,10,w])
horizontal_lines.append([1,h-10,w,h-10,w])

for h_ in range(h):
    region = raw_img[h_, 0:w]
    #count the pixel num
    total=np.sum(region)
    if total < 110:
        gap = abs(h_ - gap_lin_y)
        gap_judge = (gap< 110)
        if np.sum(gap_judge)>0:
            continue
        else:
            cv2.line(copy_img,(0,h_),(w,h_),(0,0,255),thickness=1)
            gap_lin_y=np.concatenate([gap_lin_y,np.array(h_).reshape((-1,1))],axis=0)
            horizontal_lines.append([0,h_,w,h_,w])
t2=time.time()
print('run time:',t2-t1)

# #cv2.imwrite('result234.png',copy_img)
# cv2.imshow('img',copy_img)
# key=cv2.waitKey(0)
#
# print(vertical_lines)
# print(horizontal_lines)

vertical_lines,horizontal_lines=np.array(vertical_lines),np.array(horizontal_lines)
vertical_lines=vertical_lines[vertical_lines[:,0].argsort()]
horizontal_lines=horizontal_lines[horizontal_lines[:,1].argsort()]

plot=[]

for i in range(vertical_lines.shape[0]):
    v1 = vertical_lines[i]
    x1,_,_,_,_=v1
    if i !=vertical_lines.shape[0]-1:
        v2=vertical_lines[i+1]
        x2, _, _, _, _ = v2

        for j in range(horizontal_lines.shape[0]):
            h1=horizontal_lines[j]
            _,y1,_,_,_=h1
            if j !=horizontal_lines.shape[0]-1:
                h2=horizontal_lines[j+1]
                _, y2, _, _, _ = h2
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(copy_img2,(x1,y1),(x2,y2),(0,0,255),thickness=2)
                plot.append([x1,y1,x2,y2])


cv2.imwrite('experimet.jpg',copy_img2)
print(plot)
print(len(plot))












