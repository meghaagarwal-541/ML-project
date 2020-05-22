#!/usr/bin/env python
# coding: utf-8

# In[111]:


import tensorflow as tf
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


# In[112]:


EPOCHS =20
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((150, 150))
image_size = 0
width=256
height=256
depth=3
train_dir=r"C:\Users\hp\Desktop\My website.html\MachineLearningProject\Dataset\fruits-fresh-and-rotten-for-classification\dataset\train"
valid_dir=r"C:\Users\hp\Desktop\My website.html\MachineLearningProject\Dataset\fruits-fresh-and-rotten-for-classification\dataset\test"
train_folder=listdir(train_dir)
valid_folder=listdir(valid_dir)


# In[113]:


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('acc') >= 0.96):   
            print("Reached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# In[114]:


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[115]:


callbacks = MyCustomCallback()


# In[116]:


train_image_list, train_image_label= [], []
for fruits_folder in train_folder:
    print(f"processing {fruits_folder} ...")
    fruits_img_folder= listdir(f"{train_dir}/{fruits_folder}")
    #print(disease_img_folder)
    for fruits_img in fruits_img_folder:
    #for disease_img in disease_img_folder[: : 2]:    
        image_directory = f"{train_dir}/{fruits_folder}/{fruits_img}"
        if image_directory.endswith(".png") == True or image_directory.endswith(".TIFF") == True:
            train_image_list.append(convert_image_to_array(image_directory))
            train_image_label.append(fruits_folder)
print("[INFO] Image loading completed")


# In[117]:


print(len(train_image_label))


# In[118]:


valid_image_list, valid_image_label= [], []
for fruits_folder in valid_folder:
    print(f"processing {fruits_folder} ...")
    fruits_img_folder= listdir(f"{valid_dir}/{fruits_folder}")
    for fruits_img in fruits_img_folder:
    #for disease_img in disease_img_folder[: : 2]: 
        image_directory = f"{valid_dir}/{fruits_folder}/{fruits_img}"
        if image_directory.endswith(".png") == True or image_directory.endswith(".TIFF") == True:
            valid_image_list.append(convert_image_to_array(image_directory))
            valid_image_label.append(fruits_folder)
print("[INFO] Image loading completed")


# In[119]:


print(len(valid_image_label))


# In[120]:


from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
bin_train_image_labels = label_binarizer.fit_transform(train_image_label)
bin_valid_image_labels = label_binarizer.fit_transform(valid_image_label)
pickle.dump(label_binarizer,open('Label_Instance_ped123.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


# In[121]:


print(n_classes)


# In[122]:


np_train_image_list = np.array(train_image_list, dtype=np.float32) / 255.0
np_valid_image_list = np.array(valid_image_list, dtype=np.float32) / 255.0


# In[123]:


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


# In[124]:


model = Sequential() 
model.add(Flatten(input_shape=np_train_image_list.shape[1:])) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(n_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
history = model.fit(np_train_image_list,bin_train_image_labels, epochs=EPOCHS,batch_size=BS,validation_data=(np_valid_image_list, bin_valid_image_labels))


# In[125]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(np_valid_image_list, bin_valid_image_labels)
print(f"Test Accuracy: {scores[1]*100}")


# In[127]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(np_valid_image_list, bin_valid_image_labels)
print(f" std Test Accuracy: numpy.std {scores[1]*100}")


# In[128]:


model.save("model_ped123.h5")


# In[129]:


print(history.history.keys())


# In[130]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[131]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




