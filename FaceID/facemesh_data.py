import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import json
import pyodbc
import os

# Nhap anh va DNN Caffe
net = cv2.dnn.readNetFromCaffe("F:/09.ComputerVision/project/FaceID/dnn/deploy.prototxt.txt",
                               "F:/09.ComputerVision/project/FaceID/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel")

# List anh trong thu muc bat ky
path = input("Folder path: ")
folder_path = path.replace("\\","/")
folder_list = os.listdir(folder_path)
data = dict(label=[],data=[])
for folder in folder_list:
    image_listname = os.listdir(f"{folder_path}/{folder}")
    for name in image_listname:
        # Doc anh
        image_path = os.path.join(folder_path,folder,name)
        image = cv2.imread(image_path)
        h,w = image.shape[:2]

        # Tach khuon mat khoi anh
        blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104,117,123),swapRB=False,crop=False)
        net.setInput(blob)
        faces = net.forward()
        i = np.argmax(faces[0,0,:,2])
        confident = faces[0,0,i,2]
        if confident > 0.5:
            boundary = faces[0,0,i,3:7]*np.array([w,h,w,h])
            x,y,x_end,y_end = boundary.astype("int")
            roi = image[y:y_end,x:x_end]

        # Khoi tao va tinh toan face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        rgb_image = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_image)

        # Luu toa do x,y,z cua face mesh vao list
        mesh_list = []
        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                for a in range(0,468):
                    lm = facial_landmarks.landmark[a]
                    lm_arr = np.array([lm.x,lm.y,lm.z])
                    x,y,z = lm_arr*np.array([w,h,w])
                    mesh_list.append([int(x),int(y),int(z)])
        data["label"].append(folder.upper())
        data["data"].append(json.dumps(mesh_list))

# Luu list vao SQL
df = pd.DataFrame({"PERSON_NAME":data["label"],"FEATURE":data["data"]})
database = "mssql+pyodbc://DESKTOP-50M8QJP\\SQLEXPRESS/COMPUTER_VISION?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(database)
df.to_sql("FACEDATA",engine,if_exists="append",index=False)