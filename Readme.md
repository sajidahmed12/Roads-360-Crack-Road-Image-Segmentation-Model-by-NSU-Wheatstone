# **Roads360 Image Segmentation**

### Objective
In the case of the autonomous driving in Bangladesh , given an front camera view, the car needs to know where is the road and where are the Crackes on the road surface . In this project, we trained a deep neural network to label the pixels of a road in images, by using a method named Fully Convolutional Network (FCN). In this project, FCN-VGG16 is implemented and trained with our Own collected dataset for road and Crack segmentation.

### Demo output

---
![demo_gif #link_willbe_updated ][video]



### 1 Code & Files

### 1.1 My project includes the following files and folders

* [main.py](main.py) is the main code for demos
* [testing.py](testing.py) single images testing
* [helper.py](hepler.py) includes some helper functions for preprocessing and image labeling masking
* [data](data) folder contains our labeled images of road data, the VGG model and source images.
* [model](model) folder is used to save the trained model
* [outputs](outputs) folder contains the labeled Output of model from test images



### 1.2 Dependencies & my environment

Anaconda is used for managing the environment.

* Python 3.7.3 , tensorflow-gpu v1.13, CUDA 10.0 CuDnn 7.5.0, Numpy, SciPy,glob,tdqm
* OS: Windows 10 Pro
* CPU: Intel® Core™ i7-7700 CPU @ 3.60 (GHz Base Frequency) 4.20 GHz (Max Turbo Frequency) 4 core 8 Threads
* GPU: NVidia GeForce GTX 1070 (8GB VRAM)
* HP 512 GB M.2 SATA SSD
* Memory: 16GB DDR4 @2400 MHz

### 1.3 How to run the code and Train the model

* Download our Road Images data (training and testing)
Download the [dataset](#link_will_be_given)
from [here](#link_will_be_given).  Extract the
dataset in the **data** folder. This will create the folder **data_road** with all
the training a test images.

* Load pre-trained VGG-16 model using this function ```maybe_download_pretrained_vgg()``` in ```helper.py``` file.

## Run the code: For Model Training
Open cmd in widnows or open terminal in Ubuntu/any linux distro then activate your virtual anaconda enviornmentexample :

```sh
activate tensorflow
``` 
then type

```sh
python train_model.py
```

## Run the code: For  Testing
(4) Use my pretrained model to predict with your own images. 

Download my trained model from this link [here](#_our_trained_model_gdrive_link)
and place it in the[model](model). folder

Testing  there is a python  Script called ```test_model.py ``` 

Then run the code using : 

```sh
python test_model.py
```
This will create a directory in the output folder and save all the images one by one with the predicted labels.  



#### 1.4. Version History

* 0.1.1
    * Updated documents
    * Date 7 December 2017 by [Marvin]

* 0.1.0
    * The first proper release
    * Date 6 December 2017 by [JunshengFu]

* 0.1.1 
    * Updated Model for Road Crack and Surroundings by team CSE499.AZK-Team-09  
    * on August 2019 Developed By me (Sajid Ahmed)  



## 2 Network Architecture

#### 2.1 Fully Convolutional Networks (FCN) Modified Model Architecture.

![][image0]

FCNs can be described as the above example: a pre-trained model, follow by
1-by-1 convolutions, then followed by transposed convolutions. Also, we
can describe it as **encoder** (a pre-trained model + 1-by-1 convolutions)
and **decoder** (transposed convolutions).

#### 2.2 Fully Convolutional Networks for Semantic Segmentation

![][image1]

The Semantic Segmentation network provided by this
[paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
also look into the paper by KITTISEG 
[paper2](https://arxiv.org/abs/1612.07695)
learns to combine coarse, high layer informaiton with fine, low layer
information. The pooling and prediction
layers are shown as grid that reveal relative spatial coarseness,
while intermediate layers are shown as vertical lines

* The encoder
    * VGG16 model pretrained on ImageNet for classification (see VGG16
    architecutre below) is used in encoder.
    * And the fully-connected layers are replaced by 1-by-1 convolutions.

* The decoder
    * Transposed convolution is used to upsample the input to the
     original image size.
    * Two skip connections are used in the model.

**VGG-16 architecture**

![vgg16][image2]



#### 2.3 Our Full Model Flow

![][15]

#### 2.3 Classification & Loss
we can approach training a FCN just like we would approach training a normal
classification CNN.

In the case of a FCN, the goal is to assign each pixel to the appropriate
class, and cross entropy loss is used as the loss function. We can define
the loss function in tensorflow as following commands.

```sh
logits = tf.reshape(input, (-1, num_classes))
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
```
Then, we have an end-to-end model for semantic segmentation.

### 3 Dataset

#### 3.1 Training data examples from our dataset

**Original Image**

![][image3]

**Mask/Segmented image**

![][image4]

In this project, **1000** labeled images are used as training data.
Download the [Our Road dataset](#Link_will_be_given)
from [here](#Link_will_be_given.road.zip).



### 3.2 Training and Testing data

There are **5000** testing images are processed with the trained models.
5000 frames from are a video collected by team CSE499B.AZk-09-04 arroudnd Dhaka city and other 3000 images from random places in Dhaka City.


## 4 Experiments

Some key parameters in training stage, and the traning loss and training
time for each epochs are shown in the following table.

    epochs = 500
    batch_size = 24
    learning_rate = 0.001
    num_classes = 2
    image_shape = (160, 576)

training log can be found in 
```sh
    training_log.txt file
```

THE TRAINING..............................

```sh
(tensorflow) E:\Repository\kittiseg_model_train_ours>python train_model.py
2019-08-18 01:26:24.467290: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-08-18 01:26:24.632689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.64GiB
2019-08-18 01:26:24.636794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-08-18 01:26:25.125087: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-08-18 01:26:25.126870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0
2019-08-18 01:26:25.128133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N
2019-08-18 01:26:25.129508: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6389 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
WARNING:tensorflow:From train_model.py:37: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
2019-08-18 01:26:27.154097: W tensorflow/core/graph/graph_constructor.cc:1272] Importing a graph with a lower producer version 21 into an existing graph with producer version 27. Shape inference will have run different parts of the graph with different producer versions.
WARNING:tensorflow:From C:\Users\Administrator\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\training\saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
WARNING:tensorflow:From train_model.py:56: conv2d (from tensorflow.python.layers.convolutional) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.conv2d instead.
WARNING:tensorflow:From C:\Users\Administrator\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From train_model.py:60: conv2d_transpose (from tensorflow.python.layers.convolutional) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.conv2d_transpose instead.
WARNING:tensorflow:From train_model.py:85: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See `tf.nn.softmax_cross_entropy_with_logits_v2`.

Model build successful, starting training

EPOCH 1 ...
Loss = 248.623

EPOCH 2 ...
Loss = 5.945

EPOCH 3 ...
Loss = 4.377

EPOCH 4 ...
Loss = 4.126

EPOCH 5 ...
Loss = 3.859

EPOCH 6 ...
Loss = 3.801

EPOCH 7 ...
Loss = 3.736

EPOCH 8 ...
Loss = 3.729
...............................................
EPOCH 495 ...
Loss = 0.038

EPOCH 496 ...
Loss = 0.037

EPOCH 497 ...
Loss = 0.034

EPOCH 498 ...
Loss = 0.031

EPOCH 499 ...
Loss = 0.030

EPOCH 500 ...
Loss = 0.029

Training Finished. Saving test images to: ./outputs/
All done!
```

## 5 Discussion
---

### 5.1 Good Performance

With only 1000 labeled training images, the FCN-VGG16 performs well to find
where is the road in the testing data, and the testing speed is about 7/8
fps in the computer. The model performs very well on either highway or urban driving.
Some testing examples are shown as follows:

![][s1]

![][s2]

![][s3]

![][s4]

![][s5]


#### 5.2 Limitations

Based on my test on **5000** testing images. There are two scenarios where
th current model does NOT perform well: (1) turning spot, iamges not taken in perfect angel (2)
over-exposed area .

The bad performance at the turning spots might be
caused by the fact the angel of taking an image of road that from turning spots,
because almost all the training images are taken when the car was
driving straight or turning slightly. We might be able to improve the
performance by adding more training data that are taken at the turning spots.
As for the over-exposed area, it is more challenging.  One possible
approach is to use white-balance techniques or image restoration methods
to get the correct image. The other possible approach is to add training data with more differnent angels and over-exposed scenarios, and let the network to learn
how to segment the road even under the over-expose scenarios.

**Turning spot**

![][l2]


**Over-exposed area**

![][l3]




[//]: # (Image/video References and outputs)
[image0]: ./data/Source/fcn_general.jpg
[image1]: ./data/Source/fcn.jpg
[image2]: ./data/Source/vgg16.png
[image3]: ./data/Source/origin.png
[image4]: ./data/Source/mask.png
[s1]: ./data/Source/test2.png
[s2]: ./data/Source/test4.png
[s3]: ./data/Source/test5.png
[s4]: ./data/Source/test6.png
[s5]: ./data/Source/test7.png
[l1]: ./data/Source/test8.png
[l2]: ./data/Source/test11.png
[l3]: ./data/Source/test82.png
[l4]: ./data/Source/test87.png
[15]: image_segm_final.png
[demo_gif]: ./data/Ssource/demo.gif
[video]: ./data/Source/video.gif
[Marvin]: https://github.com/MarvinTeichmann
[JunshengFu]: https://github.com/JunshengFu


if any issues please Contact me [Md Sajid Ahmed](mailto:sajid.ahmed1@northsouth.edu)
