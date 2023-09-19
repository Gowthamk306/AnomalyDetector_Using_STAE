# AnomalyDetector_Using_STAE

**Project: Anomaly Detection Using STAE (Spatio Temporal Autoencoder)**

**Overview:**
This project utilizes Spatio Temporal Autoencoders (STAE) to detect anomalies in video footage. The CUHK Avenue dataset was employed for training and testing this model. To effectively train the model, please ensure that you have video footage with a minimum frame rate of 25 frames per second (FPS) and videos up to 10 minutes in length.

**Training Process:**
1. Create a folder named "train" and place all your training videos in it, naming them sequentially as "01.mp4," "02.mp4," and so on.
2. During the training phase, the "train.py" script will generate a folder named "frames." It is essential to delete this folder before initiating another training session. If you intend to run the script just once, you can safely ignore this step.
3. Upon completion, the training process will yield a model saved as "saved_model1.h5" in the H5 file format.

**Testing Procedure:**
1. For testing the model's anomaly detection capabilities, rename your test video as "test.mp4." Note that all test videos should be in the MP4 format.
2. Run the testing script, which will open a window and begin detecting anomalies in the video stream.
3. The testing process incorporates an automated method for calculating the threshold value, which is unique to each testing dataset.

For further details or inquiries about other aspects of this project, please don't hesitate to reach out to me at "kanithigowtham3@gmail.com."

Feel free to contact me if you have any questions or need additional information about the project.
