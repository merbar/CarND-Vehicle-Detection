import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import random
from skimage.feature import hog
import csv
import sys
import pickle
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip, ImageSequenceClip
import vehicleDetectUtil as vehicleUtil


# GLOBALS
# SVM
import vehicleDetect_hogVar as hogVar

# sliding windows
#windowSizes = [96, 128, 145]
windowSizes = [96, 145]
windowOverlap = 0.75
# classifier
classifier_imgSize = 64


def process_frame(img, debug=False):
    # sliding windows creation
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    img_size = img.shape
    x_start_stop = [0, img_size[1]]
    y_start_stop = [int(img_size[0]/2), img_size[0]-32]
    windows = vehicleUtil.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, windowSizeAr=windowSizes, xy_overlap=(windowOverlap, windowOverlap))
    method = sys.argv[1]
    if method == 'cnn':
        if debug:
            print('CNN: loading model...')
        model = load_model('cnn.h5')
        if debug:
            print('CNN: extracting windows...')
        imgs = vehicleUtil.get_window_imgs(img, windows, classifier_imgSize)
        if debug:
            print('CNN: predicting...')
        pred = model.predict(imgs)
        pred_bin = np.array([x[0] for x in pred])
        pred_bin = np.where(pred_bin > 0.5, 1, 0)
    if method == 'svm':
        svc = joblib.load('svm.pkl')
        X_scaler = joblib.load('svm_scaler.pkl')
        pca = joblib.load('svm_pca.pkl')
        if debug:
            print('SVM: extracting windows...')
        imgs = vehicleUtil.get_window_imgs(img, windows, classifier_imgSize)
        '''
        print('SVM: extracting HOG features...')
        hogImg = vehicleDetectUtil.convertClrSpace(img, colorspace=hogVar.spatial_clr)
        hog_array = []
        for channel in hogVar.hog_channel:
            hog_array.append(hog(hogImg[:,:,channel], orientations=hogVar.orient, pixels_per_cell=(hogVar.pix_per_cell, hogVar.pix_per_cell), cells_per_block=(hogVar.cell_per_block, hogVar.cell_per_block), visualise=False, feature_vector=False))           
        '''
        if debug:
            print('SVM: extracting features/ predicting...')
        features = vehicleUtil.extract_features(imgs, hogArr=None, readImg=False, cspace=hogVar.spatial_clr, spatial_size=(hogVar.spatial, hogVar.spatial),
                                hist_bins=hogVar.histbin, hist_range=(0, 256), spatialFeat = hogVar.spatialFeat, histFeat = hogVar.histFeat,
                                hogFeat=hogVar.hogFeat, hog_cspace=hogVar.hog_clrspace, hog_orient=hogVar.orient, hog_pix_per_cell=hogVar.pix_per_cell, hog_cell_per_block=hogVar.cell_per_block, hog_channel=hogVar.hog_channel)
        X = np.vstack((features)).astype(np.float64)
        scaled_X = X_scaler.transform(X)
        scaled_X = pca.transform(scaled_X)
        pred_bin = svc.predict(scaled_X[:])
    if debug:
        print('plotting hot windows...')
    ind = [x for x in range(len(pred_bin)) if pred_bin[x]==1]
    hot_windows = [windows[i] for i in ind]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    window_img = vehicleUtil.draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6) 

    # Add heat to each box in box list
    heat = vehicleUtil.add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    #heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return heatmap, labels, window_img

def process_vidFrame(img):
    heatmap, labels, window_img = process_frame(img)
    #outImg = cv2.addWeighted(img, 1., heatmap, 0.3, 0.)
    out_img = vehicleUtil.convertClrSpace(window_img, 'RGB')
    return out_img

def main():
    file = sys.argv[2]
    if file.endswith('.jpg'):
        img = cv2.imread(file)
        heatmap, labels, hotWindows_img = process_frame(img, debug=True)
        label_img = vehicleUtil.draw_labeled_bboxes(np.copy(img), labels)
        label_img = vehicleUtil.convertClrSpace(label_img, colorspace='RGB')
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(hotWindows_img)
        plt.title('Hot Windows')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.subplot(133)
        plt.imshow(label_img)
        plt.title('Car Positions')
        fig.tight_layout()
        plt.show()
    else:
        #clip = VideoFileClip(file).subclip('00:00:15.00','00:00:19.00')
        #clip = VideoFileClip(file).subclip('00:00:05.00','00:00:06.00')
        #clip = VideoFileClip(file).subclip('00:00:05.00','00:00:05.50')    
        clip = VideoFileClip(file)
        proc_clip = clip.fl_image(process_vidFrame)
        proc_output = '{}_proc.mp4'.format(file.split('.')[0])
        proc_clip.write_videofile(proc_output, audio=False)
    

if __name__ == '__main__':
    main()

