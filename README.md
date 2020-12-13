# CovidBreathAnalyzer - 

Introduction - Cities with a greater number of COVID cases like Mumbai have directed patients with mild symptoms to self-isolate themselves at homes to avoid further spread and to save resources. Hospital admission is resource-intensive and the number of free ICU beds available in hospitals are limited. The patients have been asked to self-monitor themselves and seek medical attention when their condition becomes more severe. By staying at home, mildly affected patients are isolated and do not risk spreading the disease to others.
One of the critical parameters to self-monitor during home isolation is lung condition. This is difficult to do by the patient themselves and failure to diagnose failing lung condition can lead to patients not getting themselves admitted in time.

This repository has the code for a method to monitor lung conditions of patients using a simple mobile application/IOT device and a machine learning based spectrogram analysis of recorded breathing sounds. This can trigger timely alerts in case of deterioration of respiratory health. 

The repository for the mobile application prototype is at - https://github.com/chinmayi-r/CovidBreathAnalyzerApp

This repository has the files for - 
1. WavToSpectrogram.py - Code for Serverless Google Function to convert a wav file dropped in the bucket to a spectrogram image file
2. CNNModel_CovidBreathAnalyzer.ipynb - Machine learning model version
3. Utilities - 
  i) Sorting wav files according to the respiratory symptoms
  ii) classifying_spectrograms.ipynb - Classifying spectrograms according to label and move the separate directories.
  iii) Vectorizing the spectrograms for ML training
  iv) making_csv.ipynb - For training on Google AutoML, this file creates a csv with the path of each image used for training ML Model and the corresponding respiratory symptom contained in each
  
