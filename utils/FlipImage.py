import cv2
# 1-1 will be arranged to left-bottom corner
def FlipImage(img_name,status):
    assert status=='left-bottom' or status=='left-top' or status=='right-bottom' or status=='right-top'
    img=cv2.imread(img_name)
    if status =='left-bottom':
        pass
    elif status=='left-top':
        #flip base on x-axis
        img=cv2.flip(img,0)
    elif status=='right-bottom':
        # flip base on y-axis
        img=cv2.flip(img,1)
    elif status=='right-top':
        ## flip base on y=x
        img=cv2.flip(img,-1)
    return img

if __name__ == "__main__":
    img_name = '/home/kingargroo/YOLOVISION/beauty.jpg'
    status = 'right-top'
    img=FlipImage(img_name,status)
    cv2.imwrite('img.jpg',img)
