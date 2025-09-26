import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import json
import pyodbc
import os

def find_name (u,df_sql):
    score = -1
    u = np.array(u,dtype=np.float32).flatten()
    for _,row in df_sql.iterrows():
        face_mesh = json.loads(row["FEATURE"])[0]
        v = np.array(face_mesh,dtype=np.float32).flatten()
        cosine = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
        if cosine > score: # Tim khuon mat dang tin cay nhat
            score = cosine
            name = row["PERSON_NAME"]
    if score < 0.75:
        name = "Unknow"
    return name,score

# Nhap DNN Caffe
net = cv2.dnn.readNetFromCaffe("F:/09.ComputerVision/project/FaceID/dnn/deploy.prototxt.txt",
                               "F:/09.ComputerVision/project/FaceID/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel")

video = cv2.VideoCapture(0)
while video.isOpened():
    ret,frame = video.read()
    h,w = frame.shape[:2]

    # Tach khuon mat khoi frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,117,123),swapRB=False,crop=False)
    net.setInput(blob)
    faces = net.forward()
    for i in faces.shape[2]:
        confident = faces[0,0,i,2]
        if confident > 0.5:
            box = faces[0,0,i,3:7]*np.array([w,h,w,h])
            x,y,x_end,y_end = box.astype("int")
            roi = frame[y:y_end,x:x_end]

    if not ret: 
        print("Video error")
        break
        
    cv2.imshow("Video playback",frame)
    if cv2.waitKey(10)==ord("q"):
        break
video.release()
cv2.destroyWindow("Video playback")