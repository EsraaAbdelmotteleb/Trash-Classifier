# Trash-Classifier
* Use Deep learning to classify trash
* I developed an android application to classify waste into recyclable and not recyclable. I retrained the MobileNetV2 model originally trained on the ImageNet dataset. I used the TrashNet dataset that consist of 2527 images classified into glass, paper, cardboard, plastic, metal and trash. The retrained model has a tendency to overfit. To reduce overfitting, I augmented the data, used L2 regularization and used dropout. Then, I converted the retrained model into a TensorFlowLite model and incorporated it into the application. The compressed model successfully achieved 99% training accuracy with a loss of 0.02 and an 80% testing accuracy with 1.0 loss. The mobile application’s size is 42.11MB and uses, on average, 100KB of memory/hour. It takes 0.3245 seconds to classify and image.

# Related Work
Gary Thung, and Mindy Yang from Stanford University created a trash classifier (2016). They built a dataset named TrashNet that composes of 2527 waste images and classified it using an AlexNet-like architecture. It’s composed of 7*7 convolutional, Dropout, Fully connected layer, and log softmax. The network also used Adam gradient descent optimization. The authors ran 50 epochs of training with a 0.0075 learning rate and batch size of 25 with a 70/30 training/testing split and achieved a 23% training accuracy and 27% test accuracy.
Another attempt used ResNet34 architecture on the same dataset described above(Ching, 2019). The developer used learning rate of 5.13e-03 and 20 epochs. They achieved a 92.1% test accuracy and 91.4% validation accuracy.
There have been no attempts to develop a trash classifier for android phones using deep learning.

# Data description
I used the TrashNet dataset. It’s composed of images belonging to six classes: glass, paper, cardboard, plastic, metal, and trash. TrashNet consists of 2527 images: 501 glass, 594 paper, 403 cardboard, 482 plastic, 410 metal, and 137 trash.
To take the pictures, the authors placed each object on a white posterboard and used sunlight and/or room lighting. The devices used were Apple iPhone 7 Plus, Apple iPhone 5S, and Apple iPhone SE (Thung and Yang, 2016). The size of the dataset is 3.5GB. The authors then resized thedataset and compressed it resulting in a dataset of size 40.9 MB.
I created another dataset that’s 400% the size of the original TrashNet through augmentation. The augmentation method used was rotation (by 90°, 180° and 270°). The size of the augmented dataset is 122 MB

# Method description
I used transfer learning to re-train the MobileNetV2 model developed by Google, and pre-trained on the ImageNet dataset, on the Tashnet Dataset. Then, I converted the classifier to TFLite and used it in an android mobile application. My mobile application is based on TensorFlow’s Android Application Example.
I decided to retrain an already existing model because the TrashNet dataset is relatively small (2527 images). Therefore, retraining will allow me to take advantage of features learned by the MobileNetV2 model. To do so, I followed the steps described below. Those steps are adapted from TensorFlowLite’s tutorial named Recognize Flowers with TensorFlow on Android. The code for steps 1 to 6 is in the file named trashClassifier.py while the code for the android application (steps 7 and 8) is in folder named Android.
1. Set up
      * Download the TrashNet dataset
      * Rescale the images using ImageDataGenerator
      * Create the training generator
      * Specify image size, batch size and the directory of training dataset
      directory.
      * Create the validation generator
      * Save the labels into a file
2. Create the base model from the pre-trained model
      * Create an instance of MobileNetV2 without including the top layer
3. Extract features
      * Freeze the convolutional base created in step B and use it for feature extraction
      * Add a classifier as a top layer
      * Compile the model
      * Train the top-level classifier.
4. Fine tune the model
      * Un-freeze the top layers of the model
      * Train the weights of the newly unfrozen layers alongside training the weights of the top-level classifier
5. Convert to TFLite
      * Use TFLiteConverter
6. Download the converted model and labels
7. Develop the android application
8. Incorporate the Converted Model into Android Studio application (Moroney,2018)
      * Add the TensorFlowLite libraries to my app
      * Import a TensorFlow Lite interpreter
      * Create an instance of an Interpreter
      * Load it with a MappedByteBuffer
      * call the run method on the Interpreter


# Model Description
I use a modified MobileNetV2 architecture. The original MobileNetV2 architecture is based on an inverted residual structure; the input and output of the residual block are thin bottleneck layers. To filter features in the intermediate expansion layer, MobileNetV2 uses lightweight depthwise convolutions. In addition, MobileNetV2 doesn’t include non-linearities in the narrow layers (Sandler, Howard, Zhu, Zhmoginov and Chen, 2018). Figure 1 shows an overview of MobileNetV2 Architecture (Sandler and Howard, 2018)
![image](https://user-images.githubusercontent.com/37912462/114861630-7f974f80-9dbb-11eb-9dbc-7a05832b6f72.png)

In the modified version, the topmost layer of the MobileNetV2 model is replaced with a fully connected classifier. The pre-trained MobileNetV2 model (trained on the ImageNet dataset) is frozen and only the weights of the top-layer classifier get updated. After this, the top few layers of the original are unfrozen, and the model is further trained. Table 1 shows a summery of the modified model.
![image](https://user-images.githubusercontent.com/37912462/114861754-afdeee00-9dbb-11eb-8382-046492a5a3a6.png)

# Experimental Procedure
The validation, training, and testing accuracy were measured while varying the number of epochs (from 1 to 100), the learning rate(1e-3,1e-5 and 1e-7), optimizer (Adam, SGD and RMSprop) and the freezing ratio (0%, 30%, 60%, 100%).
In addition, I measured the effect of using augmentation (turn right, turn left and flip vertically) to increase the size of the dataset by 300%. The augmented dataset consists of 10108 images.
I also experimented with adding L2 regulizer (through keras.regularizers.l2) to the dense layer and the convolutional layer. In addition, I experimented with and changing the drop out ratio.
For the mobile application, I will test the amount of time it takes the application to classify an image and the amount of storage and memory it takes up.

# Results 
As shown in figure 2, increasing the number of epochs increased the training accuracy and decreased the loss until about 30 epochs. After that, increasing the number of epochs has minimal impact on the training accuracy and the loss.
![image](https://user-images.githubusercontent.com/37912462/114861996-00eee200-9dbc-11eb-82ad-6ac7afa00964.png)
Changing the learning rate yielded considerable changes in the testing accuracy and loss as shown in Table 2. All those values were calculated at number of epochs = 50, freezing ratio = 100 and optimizer = SGD.
![image](https://user-images.githubusercontent.com/37912462/114862061-0fd59480-9dbc-11eb-9040-2d935a5b9332.png)
Changing the optimizer yielded minimal change in the testing accuracy and loss as shown in Table 3. All those values were calculated at number of epochs = 50, freezing ratio = 100 and learning rate 1e-5.
![image](https://user-images.githubusercontent.com/37912462/114862084-15cb7580-9dbc-11eb-8219-4ba3adf37f47.png)
Changing the freezing ratio yielded minimal change in the testing accuracy and loss as shown in Table 3. All those values were calculated at number of epochs = 50, optimizer = SGD and learning rate 1e-5.
![image](https://user-images.githubusercontent.com/37912462/114862142-2d0a6300-9dbc-11eb-823a-1873602aa3b5.png)
Augmenting the dataset caused the average training accuracy of the last 3 epochs to increase to 0.9886 and the average loss to decrease to 0.0250. In addition, it caused the testing accuracy to increase to 0.6158 and the average loss to decrease to 2.4746.
Adding an L2 regulizer to the Dense layer yielded varying effects depending on the value of l as shown in table 5. All those values were calculated at number of epochs = 50, optimizer = SGD and learning rate 1e-5 and with dropout ratio of 0.2.
![image](https://user-images.githubusercontent.com/37912462/114862222-43b0ba00-9dbc-11eb-9bb8-39d3bd738ee8.png)
Adding an L2 regulizer to the convolutional layer yielded varying effects depending on the value of l as shown in table 6. All those values were calculated at number of epochs = 50, optimizer = SGD, learning rate 1e-5 and a regulizer at the Dense layer with l = 0.1 and with dropout ratio of 0.2.
![image](https://user-images.githubusercontent.com/37912462/114862272-51663f80-9dbc-11eb-806b-4fa468645874.png)
Changing the dropout ratio yielded varying effects as shown in table 7. All those values were calculated at number of epochs = 10, optimizer = SGD, learning rate 1e-5, a regulizer at the Dense layer with l = 0.1 and a regulizer at the convolutional layer with l = 0.35
![image](https://user-images.githubusercontent.com/37912462/114862318-5f1bc500-9dbc-11eb-80b9-5898d19a2c6f.png)
Adding another dropout layer after the dense layer had varying effects as shown in table 8. All those values were calculated at number of epochs = 10, optimizer = SGD, learning rate 1e-5, a regulizer at the Dense layer with l = 0.1, a regulizer at the convolutional layer with l = 0.35 and 0.27 dropout ratio for the first dropout layer.
![image](https://user-images.githubusercontent.com/37912462/114862363-6fcc3b00-9dbc-11eb-8ffc-069d20038a4e.png)
The mobile application classifies an image in 0.3245 seconds. Its size is 42.11MB and, on average, it uses 100KB of memory/hour. Figure 3 shows screenshots of the application.
![image](https://user-images.githubusercontent.com/37912462/114862402-7ce92a00-9dbc-11eb-9351-e8451422869a.png)

# Conclusion
The compressed model successfully achieved a 99% training accuracy with a loss of 0.02 and an 80% testing accuracy with 1.0 loss. This was achieved when using the augmented data, with number of epochs = 10, L2 regularization added to the dense layer (l = and the convolutional layer and a dropout ratio of 0.27.
The model initially performed poorly because of over fitting. Before augmenting the data and adding regulations, the model was only able to achieve 56% accuracy with 1.8 loss - even though the training accuracy was as high as 99% and the training loss was as low as 0.02. To compact the effect of overfitting, I augmented the data to increase the number of images by 300%. In addition, I added L2 regularization to the dense layer and the convolutional layer.
Adding an additional dropout layer after the dense layer reduced the training accuracy to a maximum 0.65, thus limiting the possibilities for test accuracy. Therefore, I avoided using it in the final model.
The over fitting is caused by the small dataset (2527 images) and the high noise in the dataset. Figure 4 shows a few of the images of glass in the dataset. The glass pieces are only occupying almost 30%- 50% of the images while the other 70% is noise.
![image](https://user-images.githubusercontent.com/37912462/114862568-aefa8c00-9dbc-11eb-95b4-61f418a60071.png)
