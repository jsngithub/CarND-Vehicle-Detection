## **Vehicle Detection Project**
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/image0000.png
[image2]: ./examples/image8.png
[image3]: ./examples/hog_example.jpg
[image4]: ./examples/pipeline_example1.jpg
[image5]: ./examples/pipeline_example3.jpg
[image6]: ./examples/heatmap1.jpg
[image7]: ./examples/heatmap2.jpg
[image8]: ./examples/heatmap3.jpg
[image9]: ./examples/heatmap4.jpg
[image10]: ./examples/heatmap5.jpg
[image11]: ./examples/heatmap6.jpg
[image12]: ./examples/labels1.jpg
[image13]: ./examples/labels2.jpg
[image14]: ./examples/labels3.jpg
[image15]: ./examples/labels4.jpg
[image16]: ./examples/labels5.jpg
[image17]: ./examples/labels6.jpg
[image18]: ./examples/boxes1.jpg
[image19]: ./examples/boxes2.jpg
[image20]: ./examples/boxes3.jpg
[image21]: ./examples/boxes4.jpg
[image22]: ./examples/boxes5.jpg
[image23]: ./examples/boxes6.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for HOG feature extraction is contained in the `extract_features` function in `lesson_functions.py`.  The `extract_features` function is called from `build_classifier.py`.

In `build_classifier.py` I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car Image][image1]
![Not Car Image][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![Car Image][image1]
![HOG][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on numbers that provided good predictions on the test set.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features as well as spatial and histogram features.  The features were scaled using `StandardScaler()`.  All channels in the YCrCb colorspace is used for the HOG features.  The training data is split into 80% training set and 20% test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is based on the HOG `find_cars` implementation shown in Udacity.  I chose the y limits of 400 to 656 because that is the area of the picture where cars tend to appear.  In order to search for varying window sizes, I run the `find_cars` function with many different scales and concatenate the results before moving on to the heat map thresholding.  I determined what scales to use and how much overlap by experimenting on a few images.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 7 different scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Window Search Example1][image4]
![Window Search Example2][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![Heatmap 1][image6]
![Heatmap 2][image7]
![Heatmap 3][image8]
![Heatmap 4][image9]
![Heatmap 5][image10]
![Heatmap 6][image11]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![Label 1][image12]
![Label 2][image13]
![Label 3][image14]
![Label 4][image15]
![Label 5][image16]
![Label 6][image17]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Box 1][image18]
![Box 2][image19]
![Box 3][image20]
![Box 4][image21]
![Box 5][image22]
![Box 6][image23]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Depending on the overlap of the window search, I noticed that a big car (lower part of the image) sometimes gets several small boxes drawn on it instead of one large box.  Also, when two cars get close to each other they get identified as a single large car.

If I were to pursue this project further, I would use information from previous several frames to make the detection more robust, as was done in advanced lane find.  Once a car is detected I can give heavier weight to that region in the heat detection phase, making it more likely that the car will be detected in a future frame.

Further, I would improve the window search to look for bigger cars at the bottom of the screen and smaller cars towards the middle of the screen instead of doing a brute force search with various window sizes.