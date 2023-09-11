# Extrabig_Detection
This project indulges in solving the object detection problems with extra big image (more than 10000 x 10000 pixle ) in the realm of algriculture.

We first need to perform object detection on  Unmanned Aerial Vehicles（UAV）RGB Images of a maize farmlanad, and then extracts the plots. Based on the extraction results, we are able to get the yield of maize in a specific plot.

# Stream line (Process)

* Crop the UAV image into small slice

* Label the maize seedlings using labelImg

* Train YOLOV8 object detection model

* Predict the results with cropped image

* Plot the bounding box in the big image (optionnal)

* Get the gary image

* Perspective transform

* Extracts the plots

* Get the yield of maize in a specific plot
