# Hand Gesture Recognition with OpenCV 
## Introduction 
This project is a hand gesture recognition program that uses OpenCV, Mediapipe, and Dense Neural Network (DNN) to train the dataset. In this case, I only use 3 classes, namely "Fist", "Hifive", and "None". You can add more classes if you want.
## How To Use
1. Run "take dataset.py", Select add class (1) or create new class (2), if you have not taken any data before, select (2) then press enter.
2. Enter the name of the class you want to create, for example "Hifive" then press enter.
3. Point the hand pose you want to capture at the camera, the program will automatically capture 300 landmarks data of your hand, capturing every 1 second.
4. Run "train dataset.py", the dataset will be trained in 50 epochs and 16 batch sizes, using DNN architecture.
5. Once completed, the model will be saved in .h5 format.
6. Run "test model.py" to see the reading results from the model
## Demo Video
