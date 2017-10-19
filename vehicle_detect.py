import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# load pickled data
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
color_space = dist_pickle['color_space']
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
hog_channel = dist_pickle['hog_channel']
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
spatial_feat = dist_pickle['spatial_feat']
hist_feat = dist_pickle['hist_feat']
hog_feat = dist_pickle['hog_feat']
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, draw_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    box_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            #test_features = X_scaler.transform(hog_features)    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return draw_img, box_list
    
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


ystart = 400
ystop = 656

def draw_boxes_on_cars(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, debug=False):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    box_lists = []
    scales = (0.75, 1, 1.25, 1.5, 1.75, 2, 2.25)
    out_img = np.copy(img)
      
    for scale in scales:
        out_img, box_list = find_cars(img, out_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if (len(box_list)>0): box_lists.append(box_list)
    
    if (len(box_lists) > 0):
        box_lists = np.concatenate(box_lists)
        
        # Add heat to each box in box list
        heat = add_heat(heat,box_lists)
            
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,5)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        if (debug):
            return draw_img, heatmap, out_img, labels
        else:
            return draw_img
    else:
        draw_img = np.copy(img)
        labels = (np.zeros_like(heat), 0)
        if (debug):
            return draw_img, heat, out_img, labels
        else:
            return draw_img
        
def video_process(img):
    return draw_boxes_on_cars(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
 
process_video = False

if (process_video):
    #video_in = VideoFileClip('test_video.mp4')
    video_in = VideoFileClip('project_video.mp4')
    video_out = video_in.fl_image(video_process)
    video_out.write_videofile('output_video.mp4', audio=False)
else:
    counter = 1
    images = glob.glob('./test_images/*.jpg')
    for file in images:
        img = mpimg.imread(file)
        draw_img, heatmap, out_img, labels = draw_boxes_on_cars(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, debug=True)
        fig = plt.figure(figsize=(15,10))
        plt.subplot(141)
        plt.imshow(out_img)
        plt.title('Detections')
        plt.subplot(142)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.subplot(143)
        plt.imshow(labels[0], cmap='gray')
        plt.title('Labels')
        plt.subplot(144)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        fig.tight_layout()
        img = mpimg.imsave('./examples/pipeline_example{:d}.jpg'.format(counter), out_img)
        img = mpimg.imsave('./examples/heatmap{:d}.jpg'.format(counter), heatmap, cmap='hot')
        img = mpimg.imsave('./examples/labels{:d}.jpg'.format(counter), labels[0], cmap='gray')
        img = mpimg.imsave('./examples/boxes{:d}.jpg'.format(counter), draw_img)
        counter += 1
    


























