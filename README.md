# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
In this initiative, my choice was to employ transfer learning utilizing the pretrained Resnet50 model from the torchvision library. Resnet50 is a type of convolutional neural network, encompassing 50 layers that include both convolutional and fully connected types. This model possesses approximately 25 million parameters that can be trained. Originally trained on the ImageNet dataset, the model has acquired valuable understanding and insights by processing a multitude of images. By leveraging transfer learning, we can repurpose this pre-existing knowledge to augment our model for the specific task of classifying dog breeds, without requiring an extensive dataset.

Hyperparameter	Type	             Range
Learning Rate	    Continous	    interval: [0.001, 0.1]
Batch Size	           Categorical	    Values : [32, 64, 128]
Epochs	                 Categorical	  Values: [1, 2]


- Include a screenshot of completed training jobs

![Image1](https://github.com/calvin22580/Udacity-Dog-Breed-Classification/blob/main/Screenshot/Screenshot%202023-09-14%20103850.png?raw=true)


- Logs metrics during the training process

![Image2](https://github.com/calvin22580/Udacity-Dog-Breed-Classification/blob/main/Screenshot/Screenshot%202023-09-14%20104136.png?raw=true)


- First Job

![Image3](https://github.com/calvin22580/Udacity-Dog-Breed-Classification/blob/main/Screenshot/Screenshot%202023-09-14%20104412.png?raw=true)


- Second Job

![Image4](https://github.com/calvin22580/Udacity-Dog-Breed-Classification/blob/main/Screenshot/Screenshot%202023-09-14%20104753.png?raw=true)


- The Best Hyparameter

![Image6](https://github.com/calvin22580/Udacity-Dog-Breed-Classification/blob/main/Screenshot/Screenshot%202023-09-14%20105210.png?raw=true)





## Debugging and System Profiling

Debugging the model is crucial for tracking tensor values as they traverse through the model during both training and evaluation stages. Beyond capturing and observing these tensors, SageMaker also comes with predefined rules for examining the tensors, offering valuable insights into the training and evaluation mechanisms.

For the debugging phase, I opted to focus on the "Loss Not Decreasing Rule," which keeps an eye on the rate at which the loss is reducing.

System profiling is instrumental for monitoring various system metrics such as bottlenecks, CPU and GPU utilization, among others. I employed the ProfilerReport rule to produce a report that offers statistical insights into the performance metrics of the training session.

### Results

Findings from the Graph
There's a decline in the training loss as the number of steps increases.

The training loss displays some variability, which could suggest that a larger batch size might have been more effective.

The validation loss remains relatively stable and is considerably lower than the training loss right from the outset, potentially indicating overfitting.

If the graph were misleading, several approaches could be considered to mitigate overfitting:

A smaller model like Resnet18 could potentially be a better fit than Resnet50.
The introduction of regularization methods could help in curbing overfitting on the dataset.
The model might benefit from an expanded dataset.

## Deploying the Model

### Endpoint Overview
The model being deployed is a Resnet50, originally trained on the ImageNet dataset and further refined with the dog breed classification dataset.

The model accepts an image with dimensions of (3, 224, 224) and produces an array of 133 values, each corresponding to one of the 133 different dog breeds found in the dataset.

Neither softmax nor log softmax operations are applied in the model output (these operations are exclusively part of the nn.crossentropy loss during the training phase).

To identify the model's predicted label, the maximum value among the 133 output elements is located, and its corresponding index is taken as the label.

The model underwent fine-tuning for a single epoch, with a batch size of 128 and an approximate learning rate of 0.05.


![Image7](https://github.com/calvin22580/Udacity-Dog-Breed-Classification/blob/main/Screenshot/Screenshot%202023-09-14%20110414.png?raw=true)



### Instructions to query the model

Steps to Query the Model
Utilize the Image.open() method from the PIL library to load your local image, specifying its file path. This will load the image as a PIL image object.

The image must then go through preprocessing to be ready for input into the Resnet50 network. Initially, resize the image to dimensions of (3x256x256). Subsequently, apply a center crop to adjust the image size to (3x224x224). Convert the cropped image into a tensor with values ranging between 0.0 and 1.0. Lastly, normalize the tensor using well-known mean and standard deviation values.

Once the image is preprocessed, dispatch a request to the deployed endpoint, with the processed image serving as the payload.

