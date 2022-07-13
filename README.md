# MULTIMODAL SENTIMENT ANALYSIS
Final project for the 2022 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Alex Romero Honrubia**, **Alexandra Abós Ortega**, **Oriol Algaba Birba** and **Laia Gasparin Pedraza**. 

Advised by **Gerard Gallego**.

Table of Contents
=================

  * [INTRODUCTION AND MOTIVATION](#introduction-and-motivation)
  * [DATASET](#dataset)
  * [ARCHITECTURE AND RESULTS](#architecture-and-results)
	 * [Video Model](#video-model)
	 * [Audio](#audio-model)
	 * [Multimodal](#Multimodal---Audio---Visual-Emotion-Classifier )
  * [HOW TO TRAIN THE MODEL](#how-to-train-the-model)
   * [Development setup](#development-setup)
   * [Settting up Google Cloud](#Settting-up-Google-Cloud)
  * [RESULTS](#results)
	* [Results - Video Model](#results---video-model)
   	* [Results - Video Model](#results---audio-model)
   	* [Results - Video Model](#results---multimodal-model)
  * [CONCLUSIONS](#conclusions)

---
---
## INTRODUCTION AND MOTIVATION
Video conferencing is experimenting a huge growth since the pandemic accross all industries. Video conferencing is now involving extra features as it is becoming the new working environment as well as new channel to consume professional services. A key point of the communication is the sentiment associated with the message. 

Nowadays, there are videoconferencing applications that transcript the conversations. This is useful for real-time consumption, i.e. better understanding of different accents,  as a support for deafs or for conversational voicebots. This is also useful for post-processing of the meeting, to extract notes or to analyze the conversation, i.e. recruiting interview. The transcript of a conversation would be much more complete if annotated with an emotion. 

Working, studying and sometimes meeting friends through the videoconferencing platforms made us think about this need. In addition, we were curious to learn how to combine video and audio in the same model and to deep dive on the transformer. 

## DATASET
The Ryerson Audio-Visual Database of Emotional Speech and Song ([RAVDESS](https://zenodo.org/record/1188976#.Yskil-xBxTY)) contains 7,356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only, Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only. For this project, we've only considered the Audio-Video and with speech, excluding the song videos. Files named: "Video_Speech_XX.zip", one per actor. 

Videos are recorded in a controlled environment ensuring a clear audio with no background noise and good pronunciation, and with a clear image. The person is static, always looking at the camera and with a white background. The image frames only the bust and the actors do not wear any accessory in the head, neck or ears. It is a very clean context to study the model only focusing on the audio and video information that matters without troubleshooting the cancellation of distorsions. 

Each video of our interest,  has the following metadata:
```
   {'filename': '01-01-03-02-02-02-06.mp4', 'modality': '01', 'emotion': '03', 'em_intensity': '02', 'statement': '02'}
```
The filename can directly be used to extract the data labels by parsing the third index. 

As the goal of the project is to detect basic emotions, we reduced the number of classes. Dataset is annotated with: neutral, calm, happy, sad, angry, fearful, surprise, and disgust. Our classifier only aggregates them in the following emotions: sad, happy, disgust and angry. This decision was made after not getting good results with the video model. There are emotions that were easily interchanged such as happy and surprised.  

The video files are read to extract the video and audio frames. The dimension of the video frames are: C, H, W = 3, 720, 1280. All video frames are stacked in one tensor in the Dataset class and retrieved as a batch of images. This allows the model to treat it as a sequence of images with Conv2D. 

Video frames are preprocessed. As the images only consider the bust and are always in the same position, we do a Center Crop to remove the white space and then resize the outcome. Afterwards, for the Dataloader, we've implemented a collate function to ensure the length of the video sequence is always the same by adding padding. This collate function also returns the mask so that the transformer Attention can discard those.

![alt text](report_img/Dataset.png)
## ARCHITECTURE AND RESULTS
Our solution categorizes emotions based on audio and video. Each modality goes through its own unimodal-branch and the outputs are connected to get the emotion category out of a Classification layer. 
Our development approach was. First starting with the audio and video independently to achieve as unimodal. 

![alt text](report_img/Video-Audio_Unimodel-HL.png | width=150 align="center")

When the model was fine tuned and it worked with an acceptable performance, we worked on combining both models. 

![alt text](report_img/Video-Audio_Multimodel-HL.png)

In the following sections we will go through each of it.
### Video Model
The model to extract the features from the video frames is a combination of:
 * [ResNet-18](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf): A sequence of the first 9 Convolutional Neural Network Layers of the pretrained ResNet-18. Only the first 9 layers were used to get the low-level features. 
 * Positional Encoder: As a previous step for the Transfomer to know the frames order. 
 * Transformer Encoder: 3 Encoder Layers. Each Encoder layer with 3 layers and two-head attention block. 
 * Mean normalization: Before going into the classifier, the representation of the video frames are normalized with mean 1. 
 * Fully Connected Layer: the Linear layer as a Classifier.
 * Softmax: to get the output in probabilies of each emotion class.

![alt text](report_img/Video_Unimodel-LL.png)

### Audio Model
The model to extract the features from the audio is a combination of:
 * [Wav2Vec](https://ai.facebook.com/blog/wav2vec-state-of-the-art-speech-recognition-through-self-supervision/): A pretrained speech encoder model released by Facebook AI in 2019 based on self-supervised learning. It takes raw audio as input and computes a general representation. 
 * Mean normalization: Before going into the classifier, the representation of the audio frames are normalized with mean 0. 
 * Fully Connected Layer: the Linear layer as a Classifier.
 * Softmax: to get the output in probabilies of each emotion class.

![alt text](report_img/Audio_Unimodel-LL.png)

### Multimodal - Audio-Visual Emotion Classifier 
Each unimodal model, without the Classifier Fully Connected Layer, is taken as a branch for the multimodal Model. Each audio and video representation is concatenated. The concatenation is the input for the Multimodal Classifier.
![alt text](report_img/Video-Audio_Multimodel-LL.png)
## HOW TO TRAIN THE MODEL
### Development setup
To set the development environment use the command make venv. This will create the virtual environment and install all the requirements. Also mp4 data needs to be imported into the virtual machine (the dataset is too big to be stored in github). 

To execute: 
- Make sure desired configurations are set in main.py. 
- Run in the terminal: python src/main.py 

### Settting up Google Cloud
For running the model with more computational resources, we moved it to Google Cloud infrastructure. We've run it in a Virtual Machine with the following setup:
OS: Debian GNU/Linux 10 
Image: pytorch-1-11-cu113-v20220316-debian-10
Driver: nvidia-UVM
GPU: 1 x NVIDIA Tesla V100
## RESULTS
### Results - Video Model

Several approaches were considered for video: 

1. First iteration: Model from scratch: CNNs + Transformer + Linear
2. Second iteration: ResNet18 (Frozen) + Average + Linear
3. Third iteration: ResNet18 (Frozen) + Transformer + Linear
4. Fourth iteration: ResNet18 (Last 2 children nodes trained) + Transformer + Linear
5. Fifth iteration: ResNet18 (All children nodes trained) + Transformer + Linear

#### First iteration

![alt text](report_img/video1.png)

After first iteration model did not converge. For next iteration we considered:
- Use of a pre-trained model.
- Scheduler for the Learning Rate.
- More epochs. Initially run in Google Colab which at some point would suspend the session.
- More data. Lack of storing space which was later resolved.
- Use Confusion matrix to evaluate model performance. Initially only accuracy used
- Monitor loss evolution through W&B tool.

#### Second and Third iteration

![alt text](report_img/video2.png)

![alt text](report_img/video3.png)

- “Disgust” is the class with highest rate of TP.
- “Fearful” is wrongly classifed as “Surprised”.
- “Surprised” is predicted regularly misclassifying other emotion. High FP rate.
- Pretrained ResNet18 model yields to a good initial starting benchmark.
- Adding the transformer does not yield to better results a priori.

![alt text](report_img/video_2nd_3rd_iter_curves.png)

![alt text](report_img/video_2nd_3rd_iter_conf.jpeg)

#### Fourth and Fifth iteration

![alt text](report_img/video4_5.png)

- In 4th iteration, the model is mainly classifying as “Disgust” & “Surprised” all emotions.
- In 5th iteration, the model is mainly classifying as “Happy” all emotions.
- Activating children nodes does not improve performance overall.
- Testing of a more ambitious configuration (warm up scheduler + inverse square root + more parameters to test)  was not possible due to lack of computational power and time.

![alt text](report_img/video_3rd_4rt_5th_iter_curves.png)

![alt text](report_img/video_4rt_5th_iter_conf.jpeg)

### Results - Audio Model

### Results - Multimodal Model
Considering the previous results on video and audio architectures, two final multimodal approaches were selected: 
- First iteration: 
	Video: ResNet18 (Frozen) + Transformer + Linear
	Audio: Wav2Vec (last 2 Transformers active)

- Second iteration:
	Video: ResNet18 (Frozen) + Linear
	Audio: Wav2Vec (last 2 Transformers active)
	
![alt text](report_img/all1.png)

![alt text](report_img/all2.png)

- Overall performance of the Multimodal Approach is really good in validation but worse in test.
- “Calm” emotion is wrongly classified as “Neutral”.
- “Angry” emotion is always correctly classified.


![alt text](report_img/all_good_curves.jpeg)

![alt text](report_img/all_good_conf.jpeg)

- The overoptimistic results make us suspect about an underlying issue in the model or the data structure.
- We suspect eh Repetition variable might bias the current results so we decide to run 2 final additional run stratifying even more the dataset.

#### New data partition
- Data partition has a relevant impact into the model performance.
- While the model seems to learn from the train set, it is not able to generalize to validation set.
- More ambitious architecture configuration should be further explored.

![alt text](report_img/all_new_part_curves.png)

![alt text](report_img/all_new_part_conf.jpeg)

## CONCLUSIONS

- Video and Audio models work independently. 
- Multimodal model improves overall performance by combining relevant information from both approaches. 
- A more ambitious architecture for the Transformer should be further addressed.
- Poor performance of the transformer architecture might be due to complexity and data/infrastructure limitations. 
- Audio performance is good but it could be further improved with additional fine-tuning.
- A new set of experiments should be done with the new data partition.



## References
<a id="1">[1]</a> 
K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition,” in CVPR, 2016.
