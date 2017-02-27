import numpy as np
import cv2
from skimage.feature import hog

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam


# Create thresholded binary image
def makeGrayImg(img, mask=None, clrspaceOrigin='BGR', colorspace='RGB', useChannel=0):
    '''
    Returns a grey image based on the following inputs
    - mask
    - choice of color space
    - choice of channel(s) to use
    '''
    # color space conversion
    if clrspaceOrigin == 'BGR':
        if colorspace != 'BGR':
            if colorspace == 'HSV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif colorspace == 'LUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif colorspace == 'HLS':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif colorspace == 'YUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif colorspace == 'RGB':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif colorspace == 'YCrCb':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        else: cvt_img = np.copy(img)
    # it's RGB otherwise
    else:
        if colorspace != 'RGB':
            if colorspace == 'HSV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif colorspace == 'LUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif colorspace == 'HLS':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif colorspace == 'YUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif colorspace == 'BGR':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif colorspace == 'YCrCb':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        else: cvt_img = np.copy(img)

    # isolate channel
    if colorspace != 'GRAY':
        cvt_img = cvt_img[:,:,useChannel]     

    # apply image mask
    if mask is not None:
        imgMask = np.zeros_like(cvt_img)    
        ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(imgMask, mask, ignore_mask_color)
        # returning the image only where mask pixels are nonzero
        cvt_img = cv2.bitwise_and(cvt_img, imgMask)
    return cvt_img


def convertClrSpace(img, clrspaceOrigin='BGR', colorspace='RGB'):
    # color space conversion
    if clrspaceOrigin == 'BGR':
        if colorspace != 'BGR':
            if colorspace == 'HSV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif colorspace == 'LUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif colorspace == 'HLS':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif colorspace == 'YUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif colorspace == 'RGB':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif colorspace == 'YCrCb':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        else: cvt_img = np.copy(img)
    # it's RGB otherwise
    else:
        if colorspace != 'RGB':
            if colorspace == 'HSV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif colorspace == 'LUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif colorspace == 'HLS':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif colorspace == 'YUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif colorspace == 'BGR':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif colorspace == 'YCrCb':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        else: cvt_img = np.copy(img)
    return cvt_img


def scaleImgValues(img, maxVal=None):
    if maxVal==None:
        maxVal=np.max(img)
    return np.uint8(255*img/maxVal)


def writeImg(img, outFile, binary=False):
    if binary:
        # scale to 8-bit (0 - 255)
        img = np.uint8(255*img)
    cv2.imwrite(outFile, img)
    

def get_spatial_features(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


def get_colorHist_features(img, nbins=32, bins_range=(0, 256)):
    '''
    returns feature vector of all single channel histograms of the image
    note: feature vector from greyscale image will be 1/3 the size of the feature vector of a color image
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    if img.shape[2] != 0:
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    else:
        features = channel1_hist[0]    
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, readImg = True, hogArr=None, cspace='RGB', spatial_size=(32, 32),
                    hist_bins=32, hist_range=(0, 256), spatialFeat = True, histFeat = True,
                    hogFeat=True, hog_cspace='RGB', hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel=0, hogVec = True):
    # Create a list to append feature vectors to
    features = []
    i = 0
    # Iterate through the list of images
    for file in imgs:
        spatial_features = []
        hist_features = []
        hog_features = []
        # Read in each one by one
        if readImg:
            image = cv2.imread(file)
        else:
            image = np.copy(file)
        # apply color conversion if other than 'BGR'
        if spatialFeat or histFeat:
            if cspace != 'BGR':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                elif cspace == 'RGB':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
            else: feature_image = np.copy(image)
            if spatialFeat:
                # get spatial color features
                spatial_features = get_spatial_features(feature_image, size=spatial_size)
            if histFeat:
                # get histogram features
                hist_features = get_colorHist_features(feature_image, nbins=hist_bins, bins_range=hist_range)
        if hogFeat:
            # apply color conversion if other than 'RGB'
            if hogArr == None:
                if hog_cspace != 'BGR':
                    if hog_cspace == 'HSV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    elif hog_cspace == 'LUV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                    elif hog_cspace == 'HLS':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                    elif hog_cspace == 'YUV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    elif hog_cspace == 'YCrCb':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
                    elif hog_cspace == 'RGB':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else: feature_image = np.copy(image)      

                # Call get_hog_features() with vis=False, feature_vec=True
                for channel in hog_channel:
                    hog_features.append(get_hog_features(feature_image[:,:,channel], hog_orient, 
                                                    hog_pix_per_cell, hog_cell_per_block, vis=False, feature_vec=hogVec))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = hogArr[i]
            i+=1
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

def cnn_model(ch, row, col):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(row, col, ch),
              output_shape=(row, col, ch)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(optimizer="adam", loss="mse") #binary_crossentropy
    model.compile(optimizer='adam', loss="binary_crossentropy",  metrics=['accuracy'])
    return model


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    windowSizeAr=[64], xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Initialize a list to append window positions to
    window_list = []
    for windowSize in windowSizeAr:
        xy_window = (windowSize, windowSize)
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) -1
        ny_windows = np.int(yspan/ny_pix_per_step) -1
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
    # Return the list of windows
    return window_list

def get_window_imgs(img, windows, outSize, resize=True):
    imgs = []
    for window in windows:
        if resize:
            imgs.append(cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)))
        else:
            imgs.append(img[window[0][1]:window[1][1], window[0][0]:window[1][0]])
    imgs = np.array(imgs)
    return imgs

    # Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, windowSizeAr=[64]):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



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
        cv2.rectangle(img,(bbox[0][0],bbox[0][1]-20),(bbox[0][0]+100,bbox[0][1]),(125,125,125),-1)
        cv2.putText(img, 'car {}'.format(car_number),(bbox[0][0]+5,bbox[0][1]-2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0), thickness=2)
        #cv2.putText(img, 'car {}'.format(car_number),(bbox[0][0],bbox[0][1]-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    # Return the image
    return img

def draw_labeled_carBboxes(img, cars):
    # Iterate through all detected cars
    for car_number in range(len(cars)):
        bbox = cars[car_number].getBbox()
        if bbox != None:
            cv2.rectangle(img, (bbox[0][0],bbox[0][1]), (bbox[1][0],bbox[1][1]), (0,0,255), 6)
            #cv2.rectangle(img,(bbox[0][0],bbox[0][1]-20),(bbox[0][0]+100,bbox[0][1]),(125,125,125),-1)
            #cv2.putText(img, 'car {}'.format(car_number+1),(bbox[0][0]+5,bbox[0][1]-2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0), thickness=2)
    # Return the image
    return img