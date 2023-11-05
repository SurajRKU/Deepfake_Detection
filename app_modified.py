from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
       
# %%
import shutil
import os
import cv2
import math
import cv2
from mtcnn import MTCNN
import sys, os.path
import json
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
#from efficientnet.keras import EfficientNetB0
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
np.set_printoptions(precision=2,suppress=True)
# %%
def vid_to_img(base_path,filename):
    print(filename)
    if (filename.endswith(".mp4")):
        tmp_path = os.path.join(base_path, get_filename_only(filename))
        print('Creating Directory: ' + tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        print('Converting Video to Images...')
        count = 0
        video_file = os.path.join(base_path, filename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5) #frame rate
        while(cap.isOpened()):
            frame_id = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frame_id % math.floor(frame_rate) == 0):
                print('Original Dimensions: ', frame.shape)
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, get_filename_only(filename)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
                #if(count == 2):
                #    break
        cap.release()
        print("Done!")

# %%
def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

# %%
def crop_faces(base_path,filename):
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    faces_path = os.path.join(tmp_path, 'faces')
    print('Creating Directory: ' + faces_path)
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')

    for frame in frame_images:
        print('Processing ', frame)
        detector = MTCNN()
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        print('Face Detected: ', len(results))
        count = 0
        for result in results:
            bounding_box = result['box']
            print(bounding_box)
            confidence = result['confidence']
            print(confidence)
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3  # 30% as the margin
                margin_y = bounding_box[3] * 0.3  # 30% as the margin
                x1 = int(bounding_box[0] - margin_x)
                if x1 < 0:
                    x1 = 0
                x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                y1 = int(bounding_box[1] - margin_y)
                if y1 < 0:
                    y1 = 0
                y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
                if y2 > image.shape[0]:
                    y2 = image.shape[0]
                print(x1, y1, x2, y2)
                crop_image = image[y1:y2, x1:x2]
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
            else:
                print('Skipped a face..')
    faces_number=os.listdir(faces_path)
    print(len(faces_number))
    if(len(faces_number) == 0):
        status="No Faces"
        return faces_path,status
    if(len(faces_number) < 6):# & len(faces_number) > 2):
        status="Poor Lighting"
        return faces_path,status
    else:
        status="Proceed"
        return faces_path,status
    

# %%
def predict(faces_path,faces):
    model = load_model('./best_model_DFDC40_res.h5')

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['val_accuracy'])
    count=0
    classes=np.empty([1])
    for face in faces:
        img = tf.keras.preprocessing.image.load_img(os.path.join(faces_path,face), target_size=(128, 128))
        img_tensor = tf.keras.preprocessing.image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.  

        classes =model.predict(img_tensor)
        if(classes[0][0]>0.75):
            count+=1
        if count>=2:
            return classes
            
        print(classes)
       
    return classes

# %%
def execute():
        
        np.set_printoptions(precision=8,suppress=True)
        basepath="C:/Users/"+ "ASUS/Downloads/video_selected.mp4"
        basedir=os.path.dirname(basepath)
        print(basedir)
        filename=os.path.basename(basepath)
        print(filename)
        vid_to_img(basedir,filename)
        faces_path,status = crop_faces(basedir,filename)
        print (status)
        prediction=np.array([[-1]])
        if status=="Poor Lighting":
            status=-1
            basepath_upd="C:/Users/ASUS/Downloads/video_selected"
            shutil.rmtree(basepath_upd)
            os.remove(basepath)
        elif status=="No Faces":
            status=-2
            prediction=np.array([[-2]])
            basepath_upd="C:/Users/ASUS/Downloads/video_selected"
            shutil.rmtree(basepath_upd)
            os.remove(basepath)    
        else:
            status=1
            faces=os.listdir(faces_path)
            print(faces)
            prediction=predict(faces_path,faces)
            basepath_upd="C:/Users/ASUS/Downloads/video_selected"
            shutil.rmtree(basepath_upd)
            os.remove(basepath)  
        return prediction,status
# %%

# %% 
import time
#from flaskthreads import AppContextThread
@app.route('/')
def index():
    img='static/file.svg'
    return render_template('index.html',img=img)

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/team')
def team():
    return render_template('team.html')  

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/loading/prediction',methods=['GET','POST'])
def prediction():
     if request.method=='GET':
        predictions=np.empty(1)
        time.sleep(2)
        predictions,status=execute()
        print(status)
        if(status>0):    
            value=predictions[0][0]*100#$[0][0])
            value=round(value,2)
            if(value < 1):
                value = 3.00
        elif(status==-1):
            value=-1
        else:
            value=-2
        return render_template('result.html',value=value)
        

if __name__ == "__main__":
    app.run(debug=True)
    
   
# %%
