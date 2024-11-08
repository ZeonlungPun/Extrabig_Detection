import numpy as np
from PIL import Image
from PIL import ImageEnhance
import skimage.io
import os
def StrengthContrast(cropped_img_path,contrast_strength):
    img_lists=os.listdir(cropped_img_path)
    print('***strengthing the contrasity of image *******')
    for img_list in img_lists:
        path=os.path.join(cropped_img_path,img_list)
        img=skimage.io.imread(path)
        img=Image.fromarray(img)
        enh_con = ImageEnhance.Contrast(img)
        img_contrasted = enh_con.enhance(contrast_strength)
        img_contrasted = np.array(img_contrasted)
        skimage.io.imsave(cropped_img_path,img_contrasted,check_contrast=False)
    print('***finish strengthing the contrasity of image *******')
if __name__ == '__main__':
    cropped_img_path='/home/kingargroo/corn/zhechengsmall/img1'
    img_path='/home/kingargroo/corn/zhechengsmall/img1'
    StrengthContrast(cropped_img_path,contrast_strength=1.5)