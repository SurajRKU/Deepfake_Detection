# About

This project involves building and training a deep learning-based deepfake detection model using [Python](https://www.python.org), [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org). The proposed deepfake detector is based on the state-of-the-art ResNet50 architecture initialized with pretrained ImageNet weights and trained against a massive, comprehensive set of deepfake datasets employing transfer learning approach. 


https://github.com/SurajRKU/Deepfake_Detection/assets/53537228/9abbdda8-33ed-4ad9-8a12-536ff78f0d9b


### Deepfake Datasets Used

Due to the nature of deep neural networks being data-driven, it is necessary to acquire massive deepfake datasets with various different synthesis methods in order to achieve promising results. The following deepfake datasets were used in the final model.

- [Celeb-DF](https://github.com/danmohaha/celeb-deepfakeforensics)
- [Facebook Deepfake Detection Challenge (DFDC)](https://ai.facebook.com/datasets/dfdc/)

<p align="center"><img alt="" src="https://github.com/aaronchong888/DeepFake-Detect/blob/master/img/sample_dataset.png" width="80%"></p>
<br>


#### Data Preprocessing Step 1 - Extracting frames from video input

We extract all the videos from the acquired deepfake datasets above and save them as individual frame images for further processing as inputs to the face extraction module.
In order to cater for different video qualities and to optimize for the image processing performance, the following image resizing strategies were implemented:

• If resolution of the image is greater than 1900, scale it by a factor of 1/3<br />
• If resolution of the image is lesser than 300, scale it by a factor of 2<br />
• If resolution of the image is greater than 1000 and lesser than 1900, scale it by a factor of 1/2<br />
• Resize all the frames to the same aspect ratio<br />
• Save the newly scaled image in png format as shown in the below Figure<br />

<img src="https://github.com/SurajRKU/Deepfake_Detection/blob/main/Extracted_frames_sample.png" width="500" />

#### Data Preprocessing Step 2 - Extract faces from the frames with MTCNN
We use MTCNN to extract the faces from the frames. MTCNN model is used to detect faces with certain confidence levels. If the confidence level is significant enough, it draws a bounding box around the detected face, crops it and saves it as a .png image format as shown in the below figure.<br />
<img src="https://github.com/SurajRKU/Deepfake_Detection/blob/main/Extracted_faces_from_frames.png" width="500" />

#### Step 3 - Splitting the data into training, testing and validation sets 
Here Metadata is used to extract the label associated with each video. Based on this extracted label, we move the extracted faces to two directories namely “fake” and “real” and split the data in “fake” and “real” directories into training, testing and validation sets in the ratio 8:1:1. 

#### Step 4 - Model training
Given that most of the deepfake videos are synthesized using a frame-by-frame approach, we have formulated the deepfake detection task as a binary classification problem such that it would be generally applicable to both video and image contents.<br />
ResNet is used as the backbone for the development work.The training and validation graphs for ResNet50 trained on the DFDC dataset are shown in below figures 4(a) and(b). 
<img src="https://github.com/SurajRKU/Deepfake_Detection/blob/main/Resnet50_DFDC_Results.png"  width="500" />

The training and validation graphs for ResNet50 trained on the CelebDF dataset are shown in below figures 4(c) and 4(d).<br />
<img src="https://github.com/SurajRKU/Deepfake_Detection/blob/main/Resnet50_CelebDF_Results.png"  width="500" />

Thus, given a colored square image as the input, the trained model to compute an output between 0 and 1 that indicates the probability of the input image being either deepfake (0) or pristine (1).

