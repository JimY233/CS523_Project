# CS523_Project
Jiaming Yu U72316560 jiamingy@bu.edu  
Tengzi Zhao          tengzi@bu.edu 

Slice:   https://drive.google.com/file/d/1k_9wwNmFfgwszCPKtRJVRwtwlYPvSpz1/view?usp=sharing
Or you can also see it in github: https://github.com/JimY233/CS523_Project/blob/main/Facial%20Emotion%20Recognition%20on%20FER2013.pptx

Checkpoints: https://drive.google.com/drive/folders/15tRRYTLlbrdhsmzgfBG9_foL31nIaa-O?usp=sharing
epoch_300 achieves 72.1% and epoch_160 achieves 72.4%

## Introduction
This project aims to explore facial emotion recognition. Facial emotion recognition refers to identifying expressions that convey basic emotions, which is quite important for computer to understand human beings. FER has broad research prospects in human-computer interaction and emotional computing, including human-computer interaction, emotion analysis, intelligent security, entertainment, online education, intelligent medical care, etc. For example, it can be used to conduct market research for companies and obtain users' feedback to test video games.   

However, accurate and robust FERby computer models remains challenging due to theheterogeneity of human faces and variations inimages such as different facial pose and lighting. Among all techniques for FER, deep learning models, especially Convolutional Neural Networks (CNNs) have shown great potential due to their powerful automatic feature extraction and computational efficiency.   

Therefore, we decided to reproduce a paper result to  achieves state-of-theart single-network accuracy of 73.28 % on FER2013 without using extra training data.   
<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/ranking.PNG"/></div>

Using the code in github related to the paper, we run 300 epochs training and derive 72.1%.

Then we followed the exploration in the paper to tuning the network on optimizer, scheduler and fine tuning another 50 epochs. **Code for experiment part is modified by ourselves**  
We also experiment on architecture(vgg proposed by this paper, vgg16, resnet50, efficientnetb3).

However, the result of another 50 epochs fine tuning is not quite well. And we found the reason should be overfitting. We achieve our best result of 72.4% accuracy on 160 epochs.
Therefore, we also experiment on dropout rate. And the fine tuning result of dropout 0.4 has an good improvement over 300 epochs. But the best accuracy we derived is still 72.4%

## Code

### code reference
https://github.com/usef-kh/fer  

### Our modification on the code
The code referenced is all in py files and we **reorganized** the code in jupyter notebook in our github.  
In the related paper, it discusses the influence of optimizer, scheduler, fine tuning part but did not provide with code in Github. **we modify the code to make the experiment on them.**  
Besides, we also **experiment on pre-trained vgg16,resnet50,efficientb3** to compare with the result on a varient of vgg proposed by the paper  
We **explore the dropout rate** with our modified code  
Added `plot_confusion_matrix` function to demo part  

### Code organization
Outside all the folder,  
`myvgg.ipynb` and `myvgg_second.ipynb`implements 300 epoches training on vgg varient proposed by the paper referenced. one achieves 72.1% and the other achieves 72.0% 

Under `experiment_optimizer` folder, we experiment on the influence of choice of optimizer  
Under `experiment_scheduler` folder, we experiment on the influence of choice of learning rate scheduler
Under `experiment_finetuning` folder, we experiment on the influence of fine tuning: train another 50 epoches using two different scheduler: Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR)
Under `experiment_dropoutrate` folder, we experiment on dropout rate 0.3,0.4.0.5 on the effect of fine tuning
Under `experiment_architecture` folder, we experiment on efficientnetb3, resnet50, vgg16 and vgg varient

Under `demo` folder  
`myvgg_evaluate.ipynb` evaluates the test accuracy on several epoches of `myvgg.ipynb` and thus we find that the result is kind of overfitting. The result on 160 epochs achieves 72.4% accuracy. Therefore, when we fine tuning another 50 epochs   
`myvgg_demo.ipynb` derives the top-1 accuracy, top-2 accuracy, confusion matrix and saliency map of `myvgg.ipynb` on 72.1% checkpoints after 300 epoches and also on 72.4% checkpoints after 160 epochs

### Dataset  
FER2013  
Classify facial expressions from 35,685 examples of 48x48 pixel grayscale images of faces. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear). It can be divided to training set, privacy testing set(validation set) and public testing set(testing set).  

emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}  

<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/FER2013happy.PNG"/></div>

The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

To run the code here, please create a folder `datasets/fer3013` and put `fer2013.csv` inside. I.e. The path to the dataset is `datasets/fer2013/fer2013.csv` or you can change the path in the code `get_dataloaders(path='datasets/fer2013/fer2013.csv', bs=64, augment=True)` function  

### Architecure  
In the experiment, we followed the referenced code to reproduce the result.  

VGG varient:  
<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/architecture.PNG"/></div>  
The network consists of 4 convolutional stages and 3 fully connected layers. Each of the convolutional stages contains two convolutional blocks and a max-pooling layer. The convolution block consists of a convolutional layer, a ReLU activation, and a batch normalization layer. Batch normalization is used here to speed up the learning process, reduce the internal covariance shift, and prevent gradient vanishing or explosion. The first two fully connected layers are followed by a ReLU activation. The third fully connected layer is for classification. The convolutional stages are responsible for feature extraction, dimension reduction, and non-linearity. The fully connected layers are trained to classify the inputs as described by extracted features.

Thus the difference between this model and vgg is the number of the layers. Standard VGG has 5 convolutional stages while this model has 4 convolutional stages.

Loss: cross-entropy loss

optimizer: SGD with Nesterov Momentum

learning rate optimizer: Reduce Learning Rate on Plateau (RLRP) with a initial learning rate 0.01

To prevent overfitting, dropout rate 0.2 and data augmentation are used.

After 300 epoch  
We tried several times and derive 72.0% and 72.1%. It seems the accuracy changes every time maybe due to different batches selected. And perhaps the author chose the best result they derived but we did not have enough time to reach it.
 
Furthermore, the paper used another 50 epoches for fine tuning
Fine tuning used Cosine Annealing (Cosine) scheduler with a initial learning rate of 0.0001

### Code details  

**Dropout**  
We use   
```
class Vgg(VggFeatures):
    def __init__(self, drop=0.2):
        super().__init__(drop)
        self.lin3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super().forward(x)
        x = self.lin3(x)
        return x
```
to define the model and define the dropout rate inside.  

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
Under `experiment_architecture` folder    
we tried 100 epoch on efficientnetb3, vgg16, resnet50      
<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/experiment_architecture.PNG"/></div>  
As a result, vgg varient proposed by this paper works well.   


**optimizer**    
Under `experiment_optimizer` folder   
we tried 100 epoch on vgg varient proposed by the paper referenced with different optimzer: SGD, SGD with Nesterov Momentum, Average SGD, Adam, Adam with AMSGrad, Adadelta, and Adagrad.   
We run two secenerios: one is fixed learning rate, the other is with RLRP learning rate scheduler    
<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/optimizer.PNG"/></div>  
As a result, SGD with Nesterov Momentum works the best as the paper shows    


**scheduler**    
Under `experiment_scheduler` folder    
We also tried 100 epoch on vgg varient with different learning rate scheduler: Reduce Learning Rate on Plateau (RLRP),Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR), One Cycle Learning Rate (OneCycleLR), and Step Learning Rate(StepLR)     
<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/scheduler.PNG"/></div>  
The result shows RLRP works the best  


**Fine tuning and Dropout**  
Under `experiment_finetuning` and `experiment_dropoutrate` folder    
After train 300 epoches, we train another 50 epoches using two different scheduler: Cosine Annealing (Cosine), Cosine Annealing with Warm Restarts (CosineWR) and furthermore used validation data to train. We choose these two scheduler since both of these schedulers slowly oscillate the learning rate back and forth thus not allowing for major weight changes.  
This result is not quite good since I guess the result is already kind of overfitting for dropout rate 0.2  
Therefore, we also explore the effect of fine tuning when we set the vgg varient drop out rate to be 0.3 and 0.4 

<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/experiment_finetuning%26dropout.PNG"/></div>  

## Demo  
Under `demo` folder

<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/Confusion_Matrix.png"/></div>
Therefore, we can find the model works better on Happy and Surprise target. This is because class imbalance. There are more happy and surprise images in FER2013 dataset.

We can also compare it with the confusion matrix in the reproduced paper and they are similiar. The confusion matrix on the paper can be found under `images` folder. Or you can check `demo_comparison.ipynb` 

By propagating the loss back to the pixel values, a saliency map can highlight the pixels which have the most impact on the loss value. It highlights the visual features the CNN can capture from the input  
<div align=center><img width='600'src="https://github.com/JimY233/CS523_Project/blob/main/images/saliencymap.PNG"/></div>
This is the saliency map we derived. As we can see, select one image from testloader and the model predicts exactly the same as the ground truth.
Using saliency map, we can see that the model is placing a large importance on facial features of the person instead of background, hair, hand and so on.


## References  
Khaireddin, Y., & Chen, Z. (2021). Facial Emotion Recognition: State of the Art Performance on FER2013. arXiv preprint arXiv:2105.03588.
