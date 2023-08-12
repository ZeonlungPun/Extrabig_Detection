import cv2
import numpy as np


class ImageTransform(object):
    def __init__(self,src_img,input_points,transform_category):
        """
        :param src_img: the input image
        :param input_points: raw points
        :param TransformCategory: Affine or Perspective
        :return: new image with ROI and correct angle
        """
        self.input_points=input_points
        self.TransformCategory=transform_category
        self.source_img=src_img

    def AffineRotate(self, angle, center=None, scale=1.0, fill=0, interpolation=cv2.INTER_LINEAR):
        h, w = self.source_img.shape[:2]

        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(self.source_img,
            matrix, (w, h), flags=interpolation,borderValue=fill)
        return rotated, matrix

    def AffineROI(self):
        #get the Minimum outer rectangle
        box = cv2.minAreaRect(self.input_points)

        # angle +:anticlockwise
        newImg, matrix = self.AffineRotate(int(box[2]), center=box[0])
        newH, newW = newImg.shape[0:2]
        matrixA = matrix[0:2, 0:2]
        matrixB = matrix[:, -1].reshape((-1, 1))

        #get the points after transform
        transPts = matrixA @ rawPoints.T + matrixB
        transPts = transPts.T
        transPts = np.array(transPts, dtype=np.int32)

        # make roi mask
        mask = np.zeros((newH, newW), dtype=np.uint8)
        cv2.fillPoly(mask, [transPts], (255), 8, 0)
        roi = cv2.bitwise_and(newImg, newImg, mask=mask)

        return roi

    def PerspectiveRoate(self):
        x0,y0 = self.input_points[0]
        x2,y2  = self.input_points[2]
        x3,y3  = self.input_points[3]

        # #get the width and height by raw points
        h=int(np.sqrt((x0-x2)**2+(y2-y0)**2))
        w=int(np.sqrt((x3-x2)**2+(y3-y2)**2))

        pst1 = np.float32(self.input_points)
        # new points (after transform)
        pst2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        mat = cv2.getPerspectiveTransform(pst1, pst2)
        roi = cv2.warpPerspective(img, mat, (w, h))

        return roi

    def main(self):
        if self.TransformCategory =='Affine':
            roi=self.AffineROI()

        elif self.TransformCategory == 'Perspective':
            roi =self.PerspectiveRoate()

        return roi


if __name__ == '__main__':
    img = cv2.imread('/home/kingargroo/arieal/xinmi.tif')
    rawPoints = np.array([[2160, 1286], [5711, 1390], [5698, 378], [23218, 1070], [22820, 10043], [1873, 9148]])
    trans=ImageTransform(src_img=img,input_points=rawPoints,transform_category='Affine')
    roi=trans.main()
    cv2.imwrite('roi5.tif', roi)

