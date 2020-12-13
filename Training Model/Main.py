#!/usr/bin/env python
# coding: utf-8

# # Create and Train Neural Network Model

# In[1]:


#Import All Required Packages
import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


#Setting up paths
trainDir= "Dataset/train"
validDir= "Dataset/valid"
testDir= "Dataset/test"


# In[3]:


#Generating Batches for training and validating
#Batches are created after preprocessing the images in folder with processing function for mobile net
trainBatch= ImageDataGenerator(rotation_range= 40, zoom_range=[0.15,1.4], preprocessing_function= tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory= trainDir, shuffle=True, target_size=(224,224), batch_size = 32)
validBatch= ImageDataGenerator(rotation_range= 40, zoom_range=[0.15,1.4], preprocessing_function= tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory= validDir, shuffle=True, target_size=(224,224), batch_size = 32)


# In[4]:


#Download and save MobileNet Model
mobile= tf.keras.applications.MobileNet()


# In[5]:


#Original Model layers and Summary
mobile.summary()


# In[6]:


#Creating custom output for mobile net model
x = mobile.layers[-6].output
output = tf.keras.layers.Dense(units=10, activation='softmax')(x)
model = tf.keras.models.Model(inputs=mobile.input, outputs=output)


# In[7]:


#New model layers and summary
model.summary()


# In[8]:


#Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics= ['accuracy'])


# In[9]:


#Fine Tune the model to predict requred classes
for i in range(0,25):
    print("Epoch Completed: {}".format(i*10))
    model.fit(x= trainBatch, validation_data= validBatch, epochs= 10, verbose= 2)

    #Save the model
    model.save('model\mainModel{}.h5'.format(i))


# # Testing Trained Model

# In[10]:


#import packages for visualizing performance of trained model with confusion matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[11]:


#load trained Model
model= tf.keras.models.load_model('model\mainModel24.h5')


# In[12]:


#initialize prediction array 
cm= np.zeros([10,10])
lables = ["Zero","One","Two","Three","Four","Five","ThumbsUp","Ok","SpiderMan","Rock"]


# In[13]:


#preprocessing the image, Same as done to create batches
def preProcess(image):
    image= np.array(image)
    resized= cv2.resize(image, (224,224))
    resized= cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    filtered= tf.keras.applications.mobilenet.preprocess_input(resized)
    reshaped= filtered.reshape(-1,224,224,3)
    return reshaped


# In[14]:


#Access all class folder images and start predicting
count = 0
for subs in os.listdir(testDir):
    print(lables[count])
    for images in os.listdir(os.path.join(testDir, subs)):
        img = cv2.imread(os.path.join(testDir, subs, images), -1)
        prosimg = preProcess(img)
        pred = model.predict(prosimg)
        cm[count][np.argmax(pred[0])] = cm[count][np.argmax(pred[0])] + 1
    count = count + 1


# In[15]:


#Array Containing Confusion Matrix
cm


# In[16]:


#Setup Confusion Matrix and display
df_cm = pd.DataFrame(cm, lables, lables)
plt.figure(figsize=(20,10))
sn.set(font_scale=1.7) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g') # font size
plt.show()


# # Convert h5 model to tflite

# In[17]:


model= tf.keras.models.load_model('model/mainModel24.h5')


# In[18]:


converter= tf.lite.TFLiteConverter.from_keras_model(model)
tf_model= converter.convert()
with open('model\mainModel.tflite', 'wb') as f:
    f.write(tf_model)


# # Testing Model with live video

# In[19]:


#The value in the bracket can change as per the camera in use
vid= cv2.VideoCapture(1)


# In[20]:


interpreter= tf.lite.Interpreter(model_path= 'model/mainModel.tflite')
interpreter.allocate_tensors()
input_details= interpreter.get_input_details()
output_details= interpreter.get_output_details()


# In[21]:


while True:
    ret, frame= vid.read()
    if ret==0:
        print("No Camera Detected, try changing port number")
        break

    shape_fr = frame.shape
    start_pt = (int(shape_fr[1]/16),int(shape_fr[0]/4))
    end_pt = (int(shape_fr[1]/2-shape_fr[1]/16),int(shape_fr[0] - shape_fr[0]/4))

    roi = frame[int(shape_fr[0]/4):int(shape_fr[0] - shape_fr[0]/4),int(shape_fr[1]/16):int(shape_fr[1]/2-shape_fr[1]/16)]
    processFrame= preProcess(roi)
    frame = cv2.rectangle(frame, start_pt, end_pt, (255,0,0), 2)
    
#     interpreter.set_tensor(input_details[0]['index'], processFrame)
#     interpreter.invoke()
#     pred= interpreter.get_tensor(output_details[0]['index']) 
    
    pred= model.predict(processFrame)
    
    gest = lables[np.argmax(pred[0])]
    cv2.putText(frame, gest, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


# In[ ]:




