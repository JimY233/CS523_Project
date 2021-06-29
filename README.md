# CS523_Project
Jiaming Yu U72316560 jiamingy@bu.edu  
Tengzi Zhao    

Slice:   

## Introduction
This project aims to explore facial emotion recognition. Facial emotion recognition refers to identifying expressions that convey basic emotions, which is quite important for computer to understand human beings. FER has broad research prospects in human-computer interaction and emotional computing, including human-computer interaction, emotion analysis, intelligent security, entertainment, online education, intelligent medical care, etc. For example, it can be used to conduct market research for companies and obtain users' feedback to test video games. However, accurate and robust FERby computer models remains challenging due to theheterogeneity of human faces and variations inimages such as different facial pose and lighting. Therefore, we reproduce a paper result to  achieves state-of-theart single-network accuracy of 73.28 % on FER2013 without using extra training data. 
So far we derive an accuracy of 72.4% and I believe we can improve this by changing dropout rate and re-run to select different mini-batches

## Code

### code reference
https://github.com/usef-kh/fer  
The code referenced is all in py files and we reorganized the code in jupyter notebook in our github.
In the related paper, it discusses the influence of optimizer, scheduler, fine tuning and we also change the code to make the experiment on them.
Besides, we also experiment on pre-trained vgg16,resnet50,efficientb3 to compare with the result on a varient of vgg proposed by the paper

### Code organization
`myvgg.ipynb` and `myvgg_second.ipynb`implements 300 epoches training on vgg varient proposed by the paper referenced. one achieves 72.1% and the other achieves 72.0% 
`myvgg_evaluate.ipynb` evaluates the test accuracy on several epoches of `myvgg.ipynb` and thus we find that the result is kind of overfitting. The result on 160 epoches achieves 72.4% accuracy
`myvgg_demo` derives the top-1 accuracy, top-2 accuracy, confusion matrix and saliency map of `myvgg.ipynb` on 300 epoches with 72.1%

### Dataset
FER2013  
Classify facial expressions from 35,685 examples of 48x48 pixel grayscale images of faces. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear). It can be divided to training set, privacy testing set(validation set) and public testing set(testing set).  

emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}  

### Architecure
In the experiment, we followed the referenced code to reproduce the result.

VGG varient:  
The network consists of 4 convolutional stages and 3 fully connected layers. Each of the convolutional stages contains two convolutional blocks and a max-pooling layer. The convolution block consists of a convolutional layer, a ReLU activation, and a batch normalization layer. Batch normalization is used here to speed up the learning process, reduce the internal covariance shift, and prevent gradient vanishing or explosion [18]. The first two fully connected layers are followed by a ReLU activation. The third fully connected layer is for classification. The convolutional stages are responsible for feature extraction, dimension reduction, and non-linearity. The fully connected layers are trained to classify the inputs as described by extracted features.

Loss: cross-entropy loss

optimizer: SGD with Nesterov Momentum

learning rate optimizer: Reduce Learning Rate on Plateau (RLRP) with a initial learning rate 0.01

To prevent overfitting, dropout rate 0.2 and data augmentation by translation are used.

After 300 epoch  
We tried several times and derive 72.0% and 72.1%. It seems the accuracy changes every time maybe due to different batches selected. And perhaps the author chose the best result they derived but we did not have enough time to reach it.
 
Furthermore, the paper used another 50 epoches for fine tuning
Fine tuning used Cosine Annealing (Cosine) scheduler with a initial learning rate of 0.0001

### Code details

**Dropout**
We use `class Vgg(VggFeatures)` to define the model and define the dropout rate inside.

**Data Augmentation**
```
train_transform = transforms.Compose([
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),

            transforms.TenCrop(40),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
        ])
```
This augmentation includes rescaling the images up to ± 20 % of its original scale, horizontally and vertically shifting the image by up to  ± 20 % of its size, and rotating it up to ± 10 degrees.  Each of the techniques is applied randomly and with a probability of 50 %.  

### Experiment
Then we set up experiments on architecture, optimizer, scheduler and fine tuning as the paper discusses

**architecture**
we tried 100 epoch on efficientnetb3, vgg16, resnet50
vgg16: 70.24%  
resnet50: 70.88%  
efn: 69.18%  
As a result, vgg varient proposed by this paper works well.

**optimizer**
we tried 100 epoch on vgg varient proposed by the paper referenced with different optimzer: SGD, SGD with Nesterov Momentum, Average SGD, Adam, Adam with AMSGrad, Adadelta, and Adagrad.
We run two secenerios: one is fixed learning rate, the other is with RLRP learning rate scheduler

As a result, SGD with Nesterov Momentum works the best as the paper shows

**scheduler**
We also tried 100 epoch on vgg varient with different learning rate scheduler: Reduce Learning Rate on Plateau (RLRP),Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR), One Cycle Learning Rate (OneCycleLR), and Step Learning Rate(StepLR) 

The result shows RLRP works the best

**Fine tuning**
After trainint 300 epoches, we examed another 50 epoches using two different scheduler: Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR) and furthermore used validation data to train.
This result is not quite good since I guess the result is already kind of overfitting for dropout rate 0.2

## Demo

<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/Confusion_Matrix.png"/></div>
Therefore, we can find the model works better on Happy and Surprise target. This is because data imbalance. There are more happy and surprise images in FER2013 dataset.

We can also compare it with the confusion matrix in the reproduced paper and they are similiar.

<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/saliencymap.PNG"/></div>
This is the saliency map we derived. As we can see, select one image from testloader and the model predicts exactly the same as the ground truth.

## References
Khaireddin, Y., & Chen, Z. (2021). Facial Emotion Recognition: State of the Art Performance on FER2013. arXiv preprint arXiv:2105.03588.
