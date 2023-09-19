import cv2
import numpy as np 
from keras.models import load_model
from PIL import Image
import imutils
import matplotlib.pyplot as plt

def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples

    return mean_distance
def calculate_threshold(data, factor):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    threshold = median + factor * mad
    return threshold

model=load_model("saved_model.h5")

cap = cv2.VideoCapture("test_video3.mp4")
print(cap.isOpened())

loss_values = []

while cap.isOpened():

    imagedump=[]
    ret,frame=cap.read()

    if ret == False:
        break

    for i in range(10):
        ret,frame=cap.read()
        if ret == False:
            break

        image = imutils.resize(frame,width=1000,height=1200)
        frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
        gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
        gray=(gray-gray.mean())/gray.std()
        gray=np.clip(gray,0,1)
        imagedump.append(gray)

    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)

    output=model.predict(imagedump)

    loss=mean_squared_loss(imagedump,output)

    loss_values.append(loss)
threshold=calculate_threshold(loss_values, 0.7)
while cap.isOpened():

    imagedump=[]
    ret,frame=cap.read()

    if ret == False:
        break

    for i in range(10):
        ret,frame=cap.read()
        if ret == False:
            break

        image = imutils.resize(frame,width=1000,height=1200)
        frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
        gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
        gray=(gray-gray.mean())/gray.std()
        gray=np.clip(gray,0,1)
        imagedump.append(gray)

    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)

    output=model.predict(imagedump)

    loss=mean_squared_loss(imagedump,output)
    if frame is None:
        print("none")
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
    if loss>threshold:
        print('Abnormal Event Detected')
        print (loss)
        cv2.putText(image,"Abnormal Event",(220,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

    cv2.imshow("video",image)

cap.release()
cv2.destroyAllWindows()
