import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import random
from skimage.feature import hog
import csv
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import decomposition
from keras import backend as K
from keras import callbacks
from keras.regularizers import l2, activity_l2
import vehicleDetectUtil as vehicleUtil


# GLOBAL
svm = False
cnn = True
# SVM
svm_dataset_size = 0
import vehicleDetect_hogVar as hogVar

# CNN
EPOCHS = 20
BATCHSIZE = 200

def generateBatchRandom(X, y, img_x, img_y):
    batchImg = np.zeros((BATCHSIZE, img_y, img_x, 3))
    batchY = np.zeros(BATCHSIZE)
    while 1:
        for i in range(BATCHSIZE):
            i_data = np.random.randint(len(X))
            img = cv2.imread(X[i_data])
            # RANDOMIZE BRIGHTNESS
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            randBright = min(0.25+np.random.uniform(), 1.2)
            hsv[:,:,2] = hsv[:,:,2] * randBright
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # FLIP
            if (np.random.uniform() > 0.5):
                img = np.fliplr(img)
            batchImg[i] = np.copy(img)
            batchY[i] = y[i_data]
        yield batchImg, batchY


def main():
    if svm:
        vehicleFolder = 'data/vehicles'
        nonVehicleFolder = 'data/non-vehicles'
    else:
        vehicleFolder = 'data/vehicles_cnn'
        nonVehicleFolder = 'data/non-vehicles_cnn'
    file_types = ('jpg', 'png')
    # read in datasets
    files_vehicle = []
    files_nonVehicle = []
    print('finding input images...')
    for ext in file_types:
        files_vehicle.extend(glob.glob('{}/**/*.{}'.format(vehicleFolder, ext), recursive=True))
        files_nonVehicle.extend(glob.glob('{}/**/*.{}'.format(nonVehicleFolder, ext), recursive=True))
    print('found {} car and {} non-car images'.format(len(files_vehicle), len(files_nonVehicle)))

    vehicleTestFolder = 'test_data/vehicles'
    nonVehicleTestFolder = 'test_data/non-vehicles'
    files_test_vehicle = []
    files_test_nonVehicle = []
    for ext in file_types:
        files_test_vehicle.extend(glob.glob('{}/**/*.{}'.format(vehicleTestFolder, ext), recursive=True))
        files_test_nonVehicle.extend(glob.glob('{}/**/*.{}'.format(nonVehicleTestFolder, ext), recursive=True))


    img = cv2.imread(files_vehicle[0])
    imgShape = img.shape

    if svm:
        t=time.time()        
        #files_vehicle_svm = random.shuffle(files_vehicle)[0:int(svm_dataset_size/2)]
        if svm_dataset_size != 0:
            print('SVM: splitting off subset of data...')
            files_vehicle_svm = random.sample(files_vehicle, int(svm_dataset_size/2))
            files_nonVehicle_svm = random.sample(files_nonVehicle, int(svm_dataset_size/2))
        else:
            files_vehicle_svm = files_vehicle
            files_nonVehicle_svm = files_nonVehicle
        print('SVM: using {} car and {} non-car images'.format(len(files_vehicle_svm), len(files_nonVehicle_svm)))
        print('SVM: preparing features...')
        car_features = vehicleUtil.extract_features(files_vehicle_svm, cspace=hogVar.spatial_clr, spatial_size=(hogVar.spatial, hogVar.spatial),
                                hist_bins=hogVar.histbin, hist_range=(0, 256), spatialFeat = hogVar.spatialFeat, histFeat = hogVar.histFeat,
                                hogFeat=hogVar.hogFeat, hog_cspace=hogVar.hog_clrspace, hog_orient=hogVar.orient, hog_pix_per_cell=hogVar.pix_per_cell, hog_cell_per_block=hogVar.cell_per_block, hog_channel=hogVar.hog_channel)
        notcar_features = vehicleUtil.extract_features(files_nonVehicle_svm, cspace=hogVar.spatial_clr, spatial_size=(hogVar.spatial, hogVar.spatial),
                                hist_bins=hogVar.histbin, hist_range=(0, 256), spatialFeat = hogVar.spatialFeat, histFeat = hogVar.histFeat,
                                hogFeat=hogVar.hogFeat, hog_cspace=hogVar.hog_clrspace, hog_orient=hogVar.orient, hog_pix_per_cell=hogVar.pix_per_cell, hog_cell_per_block=hogVar.cell_per_block, hog_channel=hogVar.hog_channel)
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        print('SVM: normalizing features...')
        X_scaler = StandardScaler().fit(X)
        #X_scaler = RobustScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        '''
        # singular value decomposition to reduce feature space
        # this is not doing anything for now
        #pca = decomposition.PCA(n_components=2500)
        pca = decomposition.PCA()
        pca.fit(scaled_X)
        scaled_X = pca.transform(scaled_X)
        '''        

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        t2=time.time()
        print('{} seconds to create {} feature vectors of size {}'.format(round(t2-t, 5), len(scaled_X), len(scaled_X[0])))
        print('SVM: splitting train/validation data...')
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

        print('SVM: Training model...')
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVM...')
        print('SVM: Saving model...')
        joblib.dump(svc, 'svm.pkl')
        joblib.dump(X_scaler, 'svm_scaler.pkl')
        #joblib.dump(pca, 'svm_pca.pkl')
        #pickle.dump(X_scaler,open('svm_scaler.pkl','wb'))
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        ############# DEBUG
        test_imgs = glob.glob('test_images/*.jpg'.format(vehicleFolder), recursive=True)
        from scipy.ndimage.measurements import label
        windowSizes = [96, 128, 145]
        windowOverlap = 0.75
        classifier_imgSize = 64
        img = cv2.imread('test_images/test2.jpg')
        img_size = img.shape
        x_start_stop = [0, img_size[1]]
        y_start_stop = [int(img_size[0]/2), img_size[0]-32]
        windows = vehicleUtil.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, windowSizeAr=windowSizes, xy_overlap=(windowOverlap, windowOverlap))

        for file in test_imgs:
            img = cv2.imread(file)
            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            print('SVM: extracting windows...')
            imgs = vehicleUtil.get_window_imgs(img, windows, classifier_imgSize)
            features = vehicleUtil.extract_features(imgs, readImg=False, cspace=hogVar.spatial_clr, spatial_size=(hogVar.spatial, hogVar.spatial),
                                    hist_bins=hogVar.histbin, hist_range=(0, 256), spatialFeat = hogVar.spatialFeat, histFeat = hogVar.histFeat,
                                    hogFeat=hogVar.hogFeat, hog_cspace=hogVar.hog_clrspace, hog_orient=hogVar.orient, hog_pix_per_cell=hogVar.pix_per_cell, hog_cell_per_block=hogVar.cell_per_block, hog_channel=hogVar.hog_channel)
            X = np.vstack((features)).astype(np.float64)
            scaled_X = X_scaler.transform(X)
            #scaled_X = pca.transform(scaled_X)
            pred_bin = svc.predict(scaled_X[:])

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
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            label_img = vehicleUtil.draw_labeled_bboxes(np.copy(img), labels)
            fig = plt.figure()
            plt.subplot(131)
            plt.imshow(window_img)
            plt.title('Hot Windows')
            plt.subplot(132)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.subplot(133)
            plt.imshow(label_img)
            plt.title('Car Positions')
            fig.tight_layout()
            plt.show()


    if cnn:
        X = np.concatenate((files_vehicle, files_nonVehicle))
        y = np.concatenate((np.ones(len(files_vehicle)), np.zeros(len(files_nonVehicle))))
        '''
        print('CNN: loading images into memory...')
        X_imgs = []
        for each in X:
            X_imgs.append(cv2.imread(each)) 
        X_imgs = np.array(X_imgs)
        '''
        print("CNN: building model...")
        model = vehicleUtil.cnn_model(3, img.shape[0], img.shape[1])
        print("CNN: starting training...")
        generator = generateBatchRandom(X, y, img.shape[1], img.shape[0])
        cbks = [callbacks.TensorBoard(log_dir='tb_log/')]
        model.fit_generator(generator, callbacks=cbks, samples_per_epoch=20000, nb_epoch=EPOCHS)
        #model.fit(X_imgs, y, nb_epoch=EPOCHS,  batch_size=BATCHSIZE, shuffle=True, validation_split=0.15)
        print("CNN: saving model...")
        modelName = 'cnn'
        fileName = '%s' % (modelName)
        model.save('{}.h5'.format(fileName))
        '''
        json_string = model.to_json()
        with open('%s.json' % fileName, "w") as json_file:
            json_file.write(json_string)
        model.save_weights('%s.h5' % fileName)
        '''


        ################ DEBUG ################
        '''
        car_test = []
        nonCar_test = []
        for each in files_test_vehicle:
            car_test.append(cv2.imread(each)) 
        for each in files_test_nonVehicle:
            nonCar_test.append(cv2.imread(each)) 
        X = np.concatenate((car_test, nonCar_test))
        y = np.hstack((np.ones(len(files_test_vehicle)), np.zeros(len(files_test_nonVehicle))))
        pred = model.predict(X)
        #print(pred)
        pred_bin = np.array([x[0] for x in pred])
        pred_bin = np.where(pred_bin > 0.5, 1, 0)
        print('This CNN predicts: ', pred_bin[:])
        print('For these labels:  ', y.astype(int)[:])
        '''
        K.clear_session() #otherwise it often errors out here. Some python/keras garbage collection issue.

if __name__ == '__main__':
    main()