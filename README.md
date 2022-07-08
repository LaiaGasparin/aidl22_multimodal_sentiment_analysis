# MULTIMODAL SENTIMENT ANALYSIS
Final project for the 2022 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Alex Romero Honrubia**, **Alexandra Abós Ortega**, **Oriol Algaba Birba** and **Laia Gasparin Pedraza**. 

Advised by **Gerard Gallego**.

Table of Contents
=================

  * [INTRODUCTION AND MOTIVATION](#introduction-and-motivation)
  * [DATASET](#dataset)
  * [ARCHITECTURE AND RESULTS](#architecture-and-results)
	 * [Video](#video-arch)
	 * [Audio](#i3d)
	 * [Multimodal](#multimodal-network)
  * [HOW TO TRAIN THE MODEL](#how-to-train-the-model)
  * [Setting the environment](#setting-the-environment)
  * [Running training scripts](#running-training-scripts)
  * [HOW TO RUN THE PROGRAM - video_processor](#how-to-run-the-program---video_processor)
  * [Installation](#installation)
	 * [Install Docker](#install-docker)
	 * [Install docker-compose](#install-docker-compose)
	 * [Create your .env file](#create-your-env-file)
	 *
  * [HOW TO RUN THE PROGRAMvideo_capture](#how-to-run-the-program)
---
---

## INTRODUCTION AND MOTIVATION
Video conferencing is experimenting a huge growth since the pandemic accross all industries. Video conferencing is now involving extra features as it is becoming the new working environment as well as new channel to consume professional services. A key point of the communication is the sentiment associated with the message. 

Nowadays, there are videoconferencing applications that transcript the conversations. This is useful for real-time consumption, i.e. better understanding of different accents,  as a support for deafs or for conversational voicebots. This is also useful for post-processing of the meeting, to extract notes or to analyze the conversation, i.e. recruiting interview. The transcript of a conversation would be much more complete if annotated with an emotion. 

Working, studying and sometimes meeting friends through the videoconferencing platforms made us think about this need. In addition, we were curious to learn how to combine video and audio in the same model and to deep dive on the transformer. 

## DATASET
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7,356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only, Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only. For this project, the Audio-Video and Video-only are considered. 
## ARCHITECTURE AND RESULTS
Our solution categorizes emotions based on speech and video. Each modality is process independently and the outputs are connected to get the emotion category. 
![alt text](Multimodal-Emotion-Models.png)


### Video
### Audio
### Miltimodal
## How to Train the Model

## Settting the environment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt



