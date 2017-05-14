
# **Behavioral Cloning Project** 



---

## Project Goal

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track without leaving the road


[//]: # (Image References)

[image1]: ./writeup_material/LeNet.png "LeNet"
[image2]: ./writeup_material/nV_model.JPG "nVidia end to end model"
[image3]: ./writeup_material/Steering_angle_count.png "Steering Count"
[image4]: ./writeup_material/T1_center_driving.jpg "T1 center Image"
[image5]: ./writeup_material/T1_recovery.jpg "T1 Recovery Image"
[image6]: ./writeup_material/T2_center_driving.jpg "T2 center Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


## Files Structure

This project includes fellowing files and can be used to run the [simulator](https://github.com/udacity/self-driving-car-sim) in autonomous mode


| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                    | Containing the script to create and train the model                  |
| `model.h5`                   | Containing a trained convolution neural network                       |
| `drive.py` | Providing model predictions for driving the car in autonomous mode |
| `video.py` | Making a .mp4 video from the image taken from the car |




## Model Architecture and Training Strategy

### 1. Model Architecture

I start with LeNet model shown below which I used in my previous Traffic Sign Regonization project. I thought this model might be appropriate because it contains convolutional layers which is able to learn the feature from the road. However, I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting and takes me a lot of time to train the model in order to lower the loss and works not very good in this project.

![alt text][image1]


To combat the overfitting, I modified the model. I end up with taking the nVidia end to end model described in this [paper](https://arxiv.org/abs/1604.07316) as a reference and build up a similar model. The model architecture from the paper is shown in the figure below and mine shows in the form below.  Two dropout layers are added to reduce overfitting problem.
 
 ![alt text][image2]
 
 
 ```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_3 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_3 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 63, 318, 16)       448       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 61, 316, 8)        1160      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 59, 314, 4)        292       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 57, 312, 2)        74        
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 28, 156, 2)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 28, 156, 2)        0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 8736)              0         
_________________________________________________________________
dense_7 (Dense)              (None, 16)                139792    
_________________________________________________________________
dense_8 (Dense)              (None, 16)                272       
_________________________________________________________________
dense_9 (Dense)              (None, 16)                272       
_________________________________________________________________
dropout_2 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 17        
=================================================================
Total params: 142,327.0
Trainable params: 142,327.0
Non-trainable params: 0.0
_________________________________________________________________
 
 ```
 
 I used Keras lambda layer to noremalize the data between -0.5 and 0.5 with mean value equal to 0. This process reduced a lot of training time and gave a much better accuracy. (model.py lines 134)

Then I crop the image from top and bottum. The final image size is (65,320,3).

My model consists convolution neural networks with 3x3 filter sizes and depths between 2 and 16 (code lines 137-143) 

During trining the model, I found out [another paper](https://arxiv.org/abs/1511.07289) , which describes that ELU activation layers has a better performance compare to RELU layers. After compare the performance by myself, I found out it not only improve speed of learning, but also the accuracy. So I choose it to introduce nonlinearity to the model (code line 137-156).

### 2. Attempts to reduce overfitting in the model

I use dropout for regularization. A dropout rate 25% layer is used in the last convolutional layer and another 50% layer is used in the last fully connected layer which helps a lot to avoid overfitting in the model(code lines 147). The model was trained and validated on different data sets to ensure that the model was not overfitting. 

### 3. Model parameter tuning

An adam optimizer is choosed, it will converge to an effective step size without tuning manually (model.py line 167). A constant 0.0001 learning rate was also tested with a similar result.

### 4. Appropriate training data

I start with the official training data provide by Udacity as my first training data.  I plot the histogram of this data as fellowing figure.

![alt text][image3]

It's clear that most of the data is straight driving. In order to get more turning data, I augmented the data with two strategies. 

1. Flip Image : I flipped the image to get 2 times more data. If the image is flipped, the steering angle is multiplied by -1.

2. LR images : Include images from left and right camera, the steering angle is fixed with a constant correction factor = 0.2 . Although the number 0.25 can be calculated from geometric equation, the value 0.2 seems works better in my case.

I also create another training data from all tracks available for 2 laps, i.e. around 8000 images per camera. 

In this data set. To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to handle the extreme situation. These images show what a recovery looks like starting from right :

![alt text][image5]

Then I repeated this process on track two and in order to get more data points. I didn't do recovery lap on track2.

![alt text][image6]


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I randomly shuffled the data set and put 20% of the data into a validation set. 

After the collection process, I had around 8000 data points. I then preprocessed this data by the BGR2RGB tansform. Image cropping is done by the Keras model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained it for 3 epochs because I found out that after 3 epochs the loss decreacing slowly. Adam optimizer is used so that manually training the learning rate wasn't necessary. The training and validation data is fed to the model using a generator in order to save memory. Data augmentation is randomly decide during the generate process.




## Result

After I trained the model with the official data set. The final step was to run the simulator to see how well the car was driving around track one(lake track). There were a few spots where the vehicle fell off the track. For example the first corner after the bridge. To improve the driving behavior in these cases, I found out that openCV takes in BGR image instead of RGB image. So the edge is not detected in this corner. I add a BGR2RGB transform to avoid this probleem.

After that, the vehicle is able to drive autonomously around the lake track without leaving the road. I also tried the model with the old version simulator which include 2nd track (mountain track). With the model trained with only lake track data, it is possible for the car to predict correctly and drive on most of the mountain track. However, the model works not so well on the 2nd track of the new simulator(jungle track) due to the slope.

I then trained the model with the data from jungle track.  At the end of the process, the model is able to drive autonomously on all three tracks.




