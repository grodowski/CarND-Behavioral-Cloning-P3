#**Behavioral Cloning**

Udacity Self-Driving Car Nanodegree. March 2017

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sim_autonomous.png
[image2]: ./steering_00444.png
[image3]: ./steering_m00444.png
[image4]: ./steering_left_015.png
[image5]: ./steering_right_m024.png
[image6]: ./hist_before_pruning.png
[image7]: ./hist_after_pruning.png
[image8]: ./loss_with_one_convolution.png
[image9]: ./perfect_loss_after_resize_to_32x32.png
[image10]: ./hist_before_after_pruning.png

![sim screenshot][image1]

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup.md ` for summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The model can be generated using different training data sets by passing
command line arguments.

| Argument | Value |
|:------:|:-----------:|
| -d | training data directory, as saved by Udacity simulator |
| -o | full output model path, defaults to 'model/model.h5' |

Example:

```sh
python model.py -d data_track2 -o model/track2.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with two 32x3x3 filters. (model.py lines 116-119)

The model includes RELU activations to introduce nonlinearity (lines 113-127), and the data is normalized in the model using a Keras lambda layer (code line 115).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road. After including a lap on Track 2 the car became even more stable on Track 1.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from something small. I first created the simplest linear model just to verify that I can load training data and get something that can output any steering angle.

I started with a combination of Flatten and two keras Dense layers. This setup managed to keep the car straight on the road for a just few seconds and was unable to generalize data, which was uncovered by both high training and validation errors.

Then I tried with a pre-built network model inspired by the NVIDIA architecture. Introducing it was a mistake, as I didn't know where to start optimizing and results were still poor. Even though training and validation errors seemed low, the network was overfitting and behaved unpredictably while testing in automous mode. Taking a step back was a natural choice.

I started experimenting with just one convolution (32 filters, 3x3), followed by 2x2 max_pooling, one dropout and two fully connected layers. It was a good start, as this model was able to successfuly drive the car to turn 4.

I played with augmentations, input image size and parameter tuning (5 epochs were enough!). Finally I added another, similar convolutional layer which helped stabilize the recoveries and was enough to complete multiple clean laps at 20mph.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 113 - 127) consisted of a convolution neural network with the following layers:

| # | Layer | Parameters |
|:-:|:------:|:-----------:|
| 0 | Input | Nx32x32x3 |
| 1 | Cropping2D | top 10px, bottom 4px |
| 2 | Lambda | | mean variance normalization |
| 3 | Conv2D | 32 filters, 3x3 |
| 4 | Activation | RELU |
| 5 | MaxPooling2D | 2x2 |
| 6 | Dropout | 0.5 keep prob |
| 7 | Conv2D | 32 filters, 3x3 |
| 8 | Activation | RELU |
| 9 | MaxPooling2D | 2x2 |
| 10 | Dropout | 0.5 keep prob |
| 11 | Flatten | |
| 12 | Fully Connected 1 | 128 |
| 13 | Activation | RELU |
| 14 | Fully Connected 2 | 16 |
| 15 | Activation | RELU |
| 16 | Output | 1 |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Then I recorded another lap with just recoveries and cornering. Finally a lap on Track 2 has been added. Here is an example of one image sample and its augmentations after resizing to `32x32x3`.

| Image | Augmentation type | Steering angle |
|:------:|:----------------:|:--------------:|
| ![center][image2] | Original | `0.044` |
| ![flip][image3] | Horizontal Flip | `-0.044` |
| ![lcam][image4] | Left Camera (stering+epsilon) | `0.15` |
| ![rcam][image5] | Right Camera (steering-epsilon) | `-0.24` |


The final preprocessing step before training was a simple pruning algorithm to improve the distribution of steering values. (model.py lines 40-59)

Steering Angle Histogram Before/After Pruning

![hist_before][image10]

After the collection and preprocessing, I had **13440** sample images, which I finally randomly shuffled and put 20% of the data into a validation set. I used this training data for training the model, while the validation set helped determine if the model was over or under fitting.

The below training/validation loss graph represents the best results I obtained after training for 5 epochs.

![loss][image9]

### Final remarks

The  **Behavioral Cloning** project raised even more interesting questions about developing neural networks. I have listed things to try next to further improve the model driving behavior as well as code quality. Currently, `model.h5` cannot handle a full lap on simulator track 2 (just first 5 - 6 turns at low speed). Track 2 includes sharp turns, lanes, hills and more difficult lighting conditions.

Things to try:

* More augmentations: image shifts, rotations, brightness (shadows)
* Collecting more data (from Track 2)
* Experiments with more convolutional layers
* NVIDIA architecture

---

The submitted `video.mp4` includes 3 clean autonomous laps on Track 1.
