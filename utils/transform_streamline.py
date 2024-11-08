import cv2,os
import numpy as np

class ImageTransform(object):
    def __init__(self,src_img,input_points,transform_category,save=True,save_path=None):
        """
        :param src_img: the input image
        :param input_points: raw points
        :param TransformCategory: Affine or Perspective
        :return: new image with ROI and correct angle
        """
        self.input_points=input_points
        self.TransformCategory=transform_category
        self.source_img=src_img
        self.save=save
        self.save_path=save_path

    def AffineRotate(self, angle, center=None, scale=1.0, fill=0, interpolation=cv2.INTER_LINEAR):
        h, w = self.source_img.shape[:2]

        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(self.source_img,matrix, (w, h), flags=interpolation,borderValue=fill)
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
        print('mat:', mat)
        roi = cv2.warpPerspective(self.source_img, mat, (w, h))



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
            self.roi=roi
        elif self.TransformCategory == 'Perspective':
            roi =self.PerspectiveROI()
            self.roi=roi
        if self.save:
            self.save_path=os.path.join(self.save_path,"transform.jpg")
            print(self.save_path)
            cv2.imwrite(self.save_path,self.roi)

        return roi





if __name__ == '__main__':
    #img = cv2.imread('/home/kingargroo/corn/out_results/M28/renze_point111.jpg',cv2.IMREAD_GRAYSCALE)
    img=cv2.imread('/home/kingargroo/corn/out_results/M14/ninjin1_point.jpg',cv2.IMREAD_GRAYSCALE)

    print(img.shape)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    rawPoints = np.array([ [1823,350] ,[9811,1536],[8311,11426],[332,10227],             ]   )
    #rawPoints = np.array([[7135,9721],[346,9358],[749,12585],[1351,1312],[1361,1050],[2593,1102],[2602,863],[3834,914],[3849,663],[5083,717],[5098,480],[6339,290],[7587,333]])
    trans=ImageTransform2(src_img=img,input_points=rawPoints,transform_category='Perspective')
    roi=trans.main()





    cv2.imwrite('ning_roi.png', roi)

    #roi2=cv2.imread('/home/kingargroo/corn/out_results/M28/renzepoint_result.png',cv2.IMREAD_REDUCED_COLOR_2)

