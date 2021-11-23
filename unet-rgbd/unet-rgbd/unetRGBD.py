# -*- coding:utf-8 -*-

import os 
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"	

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import backend as K
import cv2
from data import *
#import tensorflow.experimental.numpy as np
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import math
from PIL import Image
from matplotlib import pyplot as plt
from focal_loss import BinaryFocalLoss

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.eps1 = 4
        self.eps2 = 3
        self.htres = 100

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        imgs_mask_test = mydata.load_test_labels()
        return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def iou(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
        score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection + 1.)
        return score

    def mylossiou(self, y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        bcloss = bce(y_true, y_pred)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
        score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection + 1.)
        iouloss = 1-score
        ret = 0.9*bcloss + 0.1*iouloss
        #y_p = K.print_tensor(ret.get_shape().as_list(), message='   ret = ')
        return ret 

    def myloss(self, y_true, y_pred):
        loss=tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 5.)
        return loss

    def weightedLoss(self, originalLossFunc, weightsList):
        def lossFunc(true, pred):

            axis = -1 #if channels last 
            #axis=  1 #if channels first


            #argmax returns the index of the element with the greatest value
            #done in the class axis, it returns the class index    
            classSelectors = K.argmax(true, axis=axis) 
                #if your loss is sparse, use only true as classSelectors

            #considering weights are ordered by class, for each class
            #true(1) if the class index is equal to the weight index   
            classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]

            #casting boolean to float for calculations  
            #each tensor in the list contains 1 where ground true class is equal to its index 
            #if you sum all these, you will get a tensor full of ones. 
            classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

            #for each of the selections above, multiply their respective weight
            weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

            #sums all the selections
            #result is a tensor with the respective weight for each element in predictions
            weightMultiplier = weights[0]
            for i in range(1, len(weights)):
                weightMultiplier = weightMultiplier + weights[i]


            #make sure your originalLossFunc only collapses the class axis
            #you need the other axes intact to multiply the weights tensor
            loss = originalLossFunc(true,pred) 
            loss = loss * weightMultiplier

            return loss
        return lossFunc

    @tf.autograph.experimental.do_not_convert
    def loss_angle(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        img = y_true.numpy()
        lbl = y_pred.numpy()
        img = np.reshape(img, (512,512,1))
        lbl = np.reshape(lbl, (512,512,1))
        img = img*255
        lbl = lbl*255
        #y_t = K.print_tensor(img, message='y_true = ')
        #y_p = K.print_tensor(lbl, message='y_pred = ')
        #img = tf.keras.preprocessing.image.array_to_img(img)
        #lbl = tf.keras.preprocessing.image.array_to_img(lbl)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #lbl=cv2.cvtColor(lbl,cv2.COLOR_BGR2GRAY)

        #ret,img = cv2.threshold(img, 127, 255, 0)
        ret,lbl = cv2.threshold(lbl, 127, 255, 0)
        #y_p = K.print_tensor(lbl, message='y_pred2 = ')

        skel1 = np.zeros(img.shape, np.float32)
        skel2 = np.zeros(lbl.shape, np.float32)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        while True:
            openImg = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, openImg)
            eroded = cv2.erode(img, element)
            skel1 = cv2.bitwise_or(skel1,temp)
            img = eroded.copy()
            if cv2.countNonZero(img)==0:
                break

        skel1 = np.array(skel1 * 255, dtype = np.uint8)  
        lines = cv2.HoughLines(skel1,1,np.pi/180,100)
        if lines is None:#When no lines are found
            lines=np.zeros((1,1,2))
        tdeg = lines[:,:,1]*180/np.pi
        clustering = DBSCAN(eps=self.eps1, min_samples=1).fit(tdeg)
        clusters = defaultdict(list)
        slines1 = []
        for i,c in enumerate(clustering.labels_): # Sort Clusters into groups
            clusters[c].append(tdeg[i])
        for i,c in enumerate(clusters): # Select one candidate per cluster
            k=(max(list(clusters[i]))[0])//90
            slines1.append((k*max(list(clusters[i]))[0])+((1-k)*min(list(clusters[i]))[0]))#choose min if angle<90 or max if angle>90
        
        while True:
            openImg = cv2.morphologyEx(lbl, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(lbl, openImg)
            eroded = cv2.erode(lbl, element)
            skel2 = cv2.bitwise_or(skel2,temp)
            lbl = eroded.copy()
            if cv2.countNonZero(lbl)==0:
                break

        skel2 = np.array(skel2 * 255, dtype = np.uint8)        
        lines = cv2.HoughLines(skel2,1,np.pi/180,self.htres)
        if lines is None:#When no lines are found
            lines=np.zeros((1,1,2))
        tdeg = lines[:,:,1]*180/np.pi
        clustering = DBSCAN(eps=self.eps1, min_samples=1).fit(tdeg)
        clusters = defaultdict(list)
        slines2 = []
        for i,c in enumerate(clustering.labels_): # Sort Clusters into groups
            clusters[c].append(tdeg[i])
        for i,c in enumerate(clusters): # Select one candidate per cluster
            k=(max(list(clusters[i]))[0])//90
            slines2.append((k*max(list(clusters[i]))[0])+((1-k)*min(list(clusters[i]))[0]))#choose min if angle<90 or max if angle>90
            
        slines = slines1+slines2
        slines = np.array(slines).reshape(-1,1)
        
        clustering = DBSCAN(eps=self.eps2, min_samples=1).fit(slines)
        clusters = defaultdict(list)
        
        for i,c in enumerate(clustering.labels_): # Sort Clusters into groups
            clusters[c].append(slines[i])
        
        error = []
        for i,c in enumerate(clusters): # Select one candidate per cluster
            if (len(list(clusters[i]))) > 1:
                error.append(max(clusters[i])-min(clusters[i]))
        
        error = np.array(error)
        e = (abs(error[error!=0].mean()))/10
        if math.isnan(e):
            e = 0.0
        #e = tf.convert_to_tensor(e)
        #y_p = K.print_tensor(e.type, message='etype = ')
        return e

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 4))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        print ("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print ("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print ("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print ("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print ("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print ("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6],axis=3)
        print(up6)
        print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        print(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7],axis=3)
        print(up7)
        print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8],axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        print(up9)
        print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print ("conv9 shape:", conv9.shape)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        print(conv10)
        model = Model(inputs, conv10)

        weights = [1,5]
        #model.compile(optimizer=Adam(lr=1e-4), loss= self.weightedLoss(tf.keras.losses.BinaryCrossentropy(from_logits=True), weights), metrics=['accuracy', self.iou, self.loss_angle], run_eagerly=True)
        #model.compile(optimizer=Adam(lr=1e-4), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', self.iou, self.loss_angle], run_eagerly=True)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', self.iou, self.loss_angle], run_eagerly=True)
        #model.compile(optimizer=Adam(lr=1e-4), loss=self.loss_angle, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

        return model


    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

        #class_weights = {0: 1., 1: 2.}

        print('Fitting model...')
        history = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=40, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.savefig('accuracy.png')
        plt.close('all')
        
        plt.plot(history.history['iou'])
        plt.plot(history.history['val_iou'])
        plt.title('model iou')
        plt.ylabel('iou')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.savefig('iou.png')
        plt.close('all')

        plt.plot(history.history['loss_angle'])
        plt.plot(history.history['val_loss_angle'])
        plt.title('angle error')
        plt.ylabel('angle error')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.savefig('angle.png')
        plt.close('all')

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close('all')

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)
 
        #model.evaluate( imgs_test,imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        piclist = []
        for line in open("./results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = imgs[i]
            img = array_to_img(img)
            img.save(path)
            cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv_pic = cv2.resize(cv_pic,(512,512),interpolation=cv2.INTER_CUBIC)
            binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, cv_save)

if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    model.summary()
    #plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()
