import cv2
import os
import streamlit as st
import numpy as np
from pygame import mixer
import time
import torch
import torch.nn as nn
import sys
# from streamlit_webrtc import webrtc_streamer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*18*18, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x = x.double()
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = x.view(-1, 64*18*18)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.softmax(x, dim=1)
        return output
def main():
    mixer.init()
    sound = mixer.Sound('alarm.wav')

    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

    lbl=['Close','Open']
    model=CNN()
    state_dict=torch.load('models/cnncat21.pt')
    model.load_state_dict(state_dict)
    model.eval()
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    placeholder=st.empty()

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = torch.from_numpy(r_eye).float()
            r_eye = r_eye.permute(2, 0, 1)
            r_eye = r_eye.unsqueeze(0)
            rpred = model(r_eye)
            rpred = torch.argmax(rpred, dim=1).cpu().numpy()
            if(rpred[0]==1):
                lbl='Open' 
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = torch.from_numpy(l_eye).float()
            l_eye = l_eye.permute(2, 0, 1)
            l_eye = l_eye.unsqueeze(0)
            lpred = model(l_eye)
            lpred = torch.argmax(lpred, dim=1).cpu().numpy()

            if(lpred[0]==1):
                lbl='Open'   
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
            
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            # st.video(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            try:
                sound.play()
                
            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.imshow('frame',frame)
        # st.image(frame,channels='RGB')
        # stop_button=st.button('Stop', key='stop_b')
        # if stop_button:
        #     display()
            

        
        placeholder.image(frame)
        
        
            
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    cv2.destroyAllWindows()

# def display():
#     sys.exit()

if __name__=='__main__':
    st.title("AlertifyMe!")
    if st.button("Start",key='start_button'):
        
        if st.button("Stop",key='stop_button'):
            # st.subheader("Thank you! You are awake now")
            st.stop()
        main()

    
        
            
