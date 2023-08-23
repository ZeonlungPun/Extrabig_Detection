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
        # get the Minimum outer rectangle
        box = cv2.minAreaRect(self.input_points)

        # angle +:anticlockwise
        newImg, matrix = self.AffineRotate(int(box[2]), center=box[0])
        newH, newW = newImg.shape[0:2]
        matrixA = matrix[0:2, 0:2]
        matrixB = matrix[:, -1].reshape((-1, 1))

        # get the points after transform
        transPts = matrixA @ rawPoints.T + matrixB
        transPts = transPts.T
        transPts = np.array(transPts, dtype=np.int32)

        # make roi mask
        mask = np.zeros((newH, newW), dtype=np.uint8)
        cv2.fillPoly(mask, [transPts], (255), 8, 0)
        roi = cv2.bitwise_and(newImg, newImg, mask=mask)

        return roi

    def PerspectiveROI(self):
        #get the Minimum outer rectangle
        box = cv2.minAreaRect(self.input_points)
        box=cv2.boxPoints(box)
        #print(box)
        # get 4 coordinates
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])

        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

        coordinates = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                             [right_point_x, right_point_y]])

        # reorder four coordinates
        coordinates=self.reorder(coordinates)
        x0,y0=coordinates[0]
        x2,y2=coordinates[2]
        x3,y3=coordinates[3]
        # #get the width and height by raw points
        h = int(np.sqrt((x0 - x2) ** 2 + (y2 - y0) ** 2))
        w = int(np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))

        pst1 = np.float32(coordinates)
        # new points (after transform)
        pst2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        mat = cv2.getPerspectiveTransform(pst1, pst2)
        roi = cv2.warpPerspective(img, mat, (w, h))



        if len(self.input_points)>4:
            inputs = np.concatenate([self.input_points, np.ones((self.input_points.shape[0], 1))], axis=1)
            print(inputs)
            outs=mat @ inputs.T
            outs=outs.T
            out_points=outs[:,0:2]/outs[:,2].reshape((-1,1))
            out_points=np.array(out_points,dtype=np.int32)
            newH, newW = roi.shape[0:2]
            # make roi mask
            mask = np.zeros((newH, newW), dtype=np.uint8)
            cv2.fillPoly(mask, [out_points], (255), 8, 0)
            roi = cv2.bitwise_and(roi, roi, mask=mask)



        return roi


    @staticmethod
    def reorder(myPoints):
        """
        reorder four coordinates to :  left top 0 ,right top 1, left bottom 2, right bottom 3
        :param myPoints: disorder points
        :return: ordered points
        """

        myNewPoints = np.zeros_like(myPoints)
        myPoints = myPoints.reshape((4, 2))
        add = myPoints.sum(1)
        myNewPoints[0] = myPoints[np.argmin(add)]
        myNewPoints[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myNewPoints[1] = myPoints[np.argmin(diff)]
        myNewPoints[2] = myPoints[np.argmax(diff)]
        return myNewPoints

    def PerspectiveRoate(self):
        # first order point left top , second order right top , third  left bottom
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
            roi =self.PerspectiveROI()

        return roi


if __name__ == '__main__':
    img = cv2.imread('/home/kingargroo/corn/out_results/M26/wenshang1_point.jpg')
    #print(img.shape)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    #rawPoints = np.array([   [315,477],[5600,382], [5824,12269],[538, 12354]    ])
    #rawPoints=np.array([[635,1274],[1350,1311],[1366,1057], [2594,1111], [2600,862], [3836,920] ,[3859,664],[5082,718],[5095,481] ,[6326,522] ,[6336,298] ,[7584,335] ,[7134,9722] ,[248,9351]     ])
    #rawPoints=np.array([ [66,132],[1706,119],[124,3704] ,[1772,3687]  ])
    rawPoints=np.array([ [501,795],[1728,799], [1738,551] ,[4213,562] ,[4221,312] ,[5446,328], [5350,10938] ,[4115,10929] ,[4135,11177],[1640,11159],[1630,11411],[377,11398]        ])
    trans=ImageTransform(src_img=img,input_points=rawPoints,transform_category='Perspective')
    roi=trans.main()
    #print(roi.shape)
    cv2.imwrite('/home/kingargroo/corn/out_results/M26/wenshang_result.jpg', roi)

