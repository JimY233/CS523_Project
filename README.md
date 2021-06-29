# CS523_Project
Jiaming Yu U72316560 jiamingy@bu.edu  
Tengzi Zhao    

Slice:   

## Code
This project aims to explore facial emotion recognition. Facial emotion recognition, for example, can be used to conduct market research for companies and obtain users' feedback to test video games. However, accurate and robust FERby computer models remains challenging due to theheterogeneity of human faces and variations inimages such as different facial pose and lighting. Therefore, we reproduce a paper result to  achieves state-of-theart single-network accuracy of 73.28 % on FER2013 without using extra training data.  

### code reference
https://github.com/usef-kh/fer  
The code referenced is all in py files and we reorganized the code in jupyter notebook

### Code organization

### Dataset
FER2013  
Classify facial expressions from 35,685 examples of 48x48 pixel grayscale images of faces. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).  

### Code details
In the experiment, we followed the referenced code to reproduce the result.

VGG varient:  
The network consists of 4 convolutional stages and 3 fully connected layers. Each of the convolutional stages contains two convolutional blocks and a max-pooling layer. The convolution block consists of a convolutional layer, a ReLU activation, and a batch normalization layer. Batch normalization is used here to speed up the learning process, reduce the internal covariance shift, and prevent gradient vanishing or explosion [18]. The first two fully connected layers are followed by a ReLU activation. The third fully connected layer is for classification. The convolutional stages are responsible for feature extraction, dimension reduction, and non-linearity. The fully connected layers are trained to classify the inputs as described by extracted features.

Loss: cross-entropy loss

optimizer: SGD with Nesterov Momentum

learning rate optimizer: Reduce Learning Rate on Plateau (RLRP) with a initial learning rate 0.01

To prevent overfitting, dropout rate 0.2 and data augmentation by translation are used.

After 300 epoch  
We tried several times and derive 71.5% and 72.2%. It seems the accuracy changes every time maybe due to different batches selected. And perhaps the author chose the best result they derived but we did not have enough time to reach it.
 
Furthermore, the paper used another 50 epoches for fine tuning
Fine tuning used Cosine Annealing (Cosine) scheduler with a initial learning rate of 0.0001

### Experiment
Then we set up experiments on architecture, optimizer, scheduler and fine tuning as the paper discusses

**architecture**
we tried 100 epoch on efficientnetb0, vgg16, resnet50

As a result, vgg works the best on validation set and test set. We also compare vgg16 and vgg varient proposed by the paper referenced
**optimizer**
we tried 100 epoch on vgg varient proposed by the paper referenced with different optimzer: SGD, SGD with Nesterov Momentum, Average SGD, Adam, Adam with AMSGrad, Adadelta, and Adagrad.
We run two secenerios: one is fixed learning rate, the other is with RLRP learning rate scheduler

As a result, SGD with Nesterov Momentum works the best as the paper shows

**scheduler**
We also tried 100 epoch on vgg varient with different learning rate scheduler: Reduce Learning Rate on Plateau (RLRP),Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR), One Cycle Learning Rate (OneCycleLR), and Step Learning Rate(StepLR) 

The result shows RLRP works the best

**Fine tuning**
After trainint 300 epoches, we examed another 50 epoches using two different scheduler: Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR) and furthermore used validation data to train

## Demo

## References
Khaireddin, Y., & Chen, Z. (2021). Facial Emotion Recognition: State of the Art Performance on FER2013. arXiv preprint arXiv:2105.03588.
