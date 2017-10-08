#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/samples1.png "Some training samples and labels"
[image2]: ./examples/Histogram_of_traindata.png "Histogram of training data"
[image3]: ./web_images/1-30kmh.jpg "Traffic Sign 1"
[image4]: ./web_images/13-yield.jpg "Traffic Sign 2"
[image5]: ./web_images/14-stop.jpg "Traffic Sign 3"
[image6]: ./web_images/23-sliproad.jpg "Traffic Sign 4"
[image7]: ./web_images/37-left_or_straight.jpg "Traffic Sign 5"
[image8]: ./examples/predictions.png "Prediction results"
[image9]: ./examples/5-softmaxes.png "5 softmax resutls"


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

The following picture shows some samples of the training data with correct labels and the statistic of labels in training data

![alt text][image1]

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

For pre-processing step, I adapt the simple method as the guideline suggest, which is to modified the value as '(pixel - 128.)/128'
This step makes the pixel values in range [-1, 1]. This technique is quick and since I don't have much time, this does the job. 




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model, which is basically LeNet, consisted of the following layers:

| Layer         		|     Description	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							    | 
| Convolution 3x3     	| 1x1 stride, `VALID` padding, outputs 28x28x6 	    |
| RELU					|												    |
| Max pooling	      	| k=2, 2x2 stride, `SAME` padding, outputs 14x14x16 |
| ReLU                  |                                                   |
| Convolution 5x5       | 1x1 stride, `VALID` padding, outputs 10x10x16     |
| ReLU                  |                                                   |
| Max pooling           | k=2, 2x2 stride, `SAME` padding, outputs 5x5x16   |
| Flatten   	        | outputs 400                                       |
| Fully connected       | outputs 120                                       |
| ReLU                  |                                                   |
| Fully connected       | outputs 84                                        |
| ReLU                  |                                                   |
| Fully connected       | outputs 43                                        |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used 40 epochs for training the model. at the beginning I used 100 epochs with learning rate 0.001, but I noticed that it only takes 20 epochs to go to 0.99 training accuracy and 30 epochs to have 1.000 training accuracy. But the test result was not as expected, so I set to 60 just to be sure and see the consistent of the model.
I used the same optimizer as LeNet lab which is AdamOptimizer.
I used learning rate at 0.001, which help the model learned get better validation accuracy since I also tried with 0.0011, 0.003, 0.002, 0.0012, 0.0015 and 0.0001.
At 0.0012 and 0.0015, the model start slower but gave more consistent result than 0.002 and 0.003. overall, learning rate at 0.001 gave the best result and consistent performance. 
I used dropout since it made higher accuracy and consistent and can reduce the number of training epochs.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results after 35 epochs were:
* training set accuracy of 99.3 %
* validation set accuracy of 95.6%
* test set accuracy of 93.9%

What architecture was chosen?
- I chose LeNet since it is known to have good result on recognizing character at low resolution and I think it would be good on signs in this project.
I made 2 modification to make it applicable to this project: 
1. Resize the input and modified the model to accept color image instead of grayscale. It looks better
2. Output length changed to 43 to match the number of labels that we are doing in this project. 

How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
- The results speak for themselves. the training accuracy after 30 epochs was nearly 99%, but the validation and test accuracy were not as expected. So I have it run 10 more and as my observation after multiple run, there is no improvement after 35th epoch. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]



they have different sizes but I did have them all resized to the size that fit the model.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image8]


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This result looks normal since in low resolution, these two sign are pretty similar.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is struggle to predict that this is a 80 km/h limit sign (probability of 0.46), since it really simillar number in low resolution. But it made a really ridiculus judgment call to predict it was a 60 km/h sigh. The top five soft max probabilities were.

![alt text][image9]


For other images, the model confidently predicts all of them with good accuracy.



