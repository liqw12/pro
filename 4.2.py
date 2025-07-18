import select
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from face import Ui_MainWindow  # 导入生成的UI类
import cv2 as cv
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
import time
import random
import os
from PIL import Image
import numpy as np 
import sqlite3
import time
from datetime import datetime
from PySide6.QtCore import QTimer
from PIL import ImageDraw,ImageFont,Image 
from unittest import result
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import time
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO)
recognizer=GestureRecognizer.create_from_options(options)
list=[]  
# def start_time_update(self):
#         # 创建定时器并连接到更新函数
#         if self.timer is None:
#             self.timer = QTimer(self)
#             self.timer.timeout.connect(self.update_time)
#             self.timer.start(1000)  # 每秒触发一次
#             self.update_time()  # 立即更新一次
    
# def update_time(self):
#         # 更新时间显示
#     current_time = datetime.now().strftime('%H:%M:%S')
#     self.timela.setText(current_time)
timer = None
ui = None  # 需要在某处初始化


def update_time():
     global timer, ui
    # 确保定时器只被创建一次
     if timer is None:
        timer = QTimer()
        # 定义时间更新函数
        def update_time():
            current_time = datetime.now().strftime('%H:%M:%S')
            ui.timela.setText(current_time) # 直接更新UI标签

# 连接定时器信号与更新函数，每秒触发一次
        timer.timeout.connect(update_time)
        timer.start(1000) # 1000毫秒 = 1秒
        update_time()
     else:
     # 如果定时器已存在，确保它处于运行状态
        if not timer.isActive():
            timer.start(1000)
# 创建全局变量
app = None
window = None
ui = None
cap=cv.VideoCapture(0)
face =cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
i=0
k=0

def creat():
    conn = sqlite3.connect('face.db')
    mess = conn.cursor()
    mess.execute("CREATE TABLE IF NOT EXISTS use (id TEXT PRIMARY KEY, name TEXT)")
    print("sucess")
    conn.close()
#插入数据
def insertSql(id,name):
    conn = sqlite3.connect('face.db')
    mess = conn.cursor()
    print(id)
    print(name)
    mess.execute("INSERT INTO use (id,name) VALUES (?, ?)", (id,name))
    conn.commit()  
    conn.close()


#进行保存图片
def save_Image():
    i=ui.id.text()
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    local=face.detectMultiScale(frame)
    for x,y,w,h in local:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi1=gray[y:y+h,x:x+w]  
        cv.imwrite("./dd/%s.jpg"%i,roi1)
        print("./dd/%s.jpg"%i)   
    
#获取图片和标签，用于训练模型
def getimageAndlabels(path):
    images=[]
    labels=[]
    #listdir查看路径下所有文件
    #这里作拼接
    imagepaths=[os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    #获得路径和路径下文件

    for imagepath in imagepaths:
      #打开图片并转换为灰色图片
        pil = Image.open(imagepath).convert("L")
        np_img=np.array(pil,dtype="uint8")
        #print(np_img.shape)
        #将人脸区域添加到images
        local= face.detectMultiScale(np_img)
        #切片路径存入lable
        id=int(os.path.split(imagepath)[1].split(".")[0])
        for x,y,w,h in local:
            images.append(np_img[y:y+h,x:x+w])
            labels.append(id) 
    return images,labels         

#训练并保存模型
def train():
    id=ui.id.text()
    rec = cv.face.LBPHFaceRecognizer.create()
    # 传入数据训练模型
    path="./dd"
    images,labels=getimageAndlabels(path)
    rec.train(images,np.array(labels))
    # #保存模型
    name=ui.lineEdit.text()
    rec.write("./trains/train.yml")


#打开相机显示到标签(用于记录训练样本)
def capture():
    while True :
        ret,frame=cap.read()
        frame=cv.flip(frame,1)
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        local=face.detectMultiScale(frame)
        for x,y,w,h in local:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi=gray[y:y+h,x:x+w]
        pixmap = convert_opencv_to_pixmap(frame)
        # 在QLabel上显示
        ui.label_2.setPixmap(pixmap)
        code = cv.waitKey(1) & 0xFF
               
     
# 用于人脸检测打卡
def faceTest():
    # x=0
    top_gesture=''
    hand_landmarks_proto=0
    id=ui.id.text()
    name=""
    recongizer=cv.face.LBPHFaceRecognizer.create()
    recongizer.read("./trains/train.yml")

    conn = sqlite3.connect('face.db')
    mess = conn.cursor()
    while True :
        ret,frame=cap.read()
        frame=cv.flip(frame,1)
        # 时间戳（毫秒）
        frame_timestamp_ms = int(time.time() * 1000)

        # OpenCV 图像（BGR）→ Mediapipe 图像（RGB）
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 在 VIDEO 模式下，使用 recognize_for_video()
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
        # print(gesture_recognition_result)

        # 处理手势识别结果
        if gesture_recognition_result.gestures:
            top_gesture = gesture_recognition_result.gestures[0][0]  # 获取第一个手势
            print(f"识别到手势: {top_gesture.category_name} (置信度: {top_gesture.score:.2f})")

        # 处理手部关键点（如果存在）
        if gesture_recognition_result.hand_landmarks:
            for hand_landmarks in gesture_recognition_result.hand_landmarks:
                # 创建 NormalizedLandmarkList 对象
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # 绘制手部关键点和连接线
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_hands = mp.solutions.hands

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        local=face.detectMultiScale(frame)
        for x,y,w,h in local:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi=gray[y:y+h,x:x+w]
            #识别摄像头人脸
            id ,conf=recongizer.predict(roi)
            label=f"ID:{id} conf:{conf:.0f}%"
            if conf < 60 and top_gesture.category_name=="Thumb_Up":
                mess.execute("select * from use where id=?",(id,))
                result=mess.fetchone()
                if result:
                    # 格式化数据为字符串
                    user = f"ID: {result[0]} | 姓名: {result[1]} |时间:{ui.timela.text()}"
                    name=result[1]
                    if id not in list:
                        # 逐行添加到QListWidget
                        ui.plainTextEdit.appendPlainText(user)
                        list.append(id) 
                       
            cv.putText(frame,label,(x,y-10),2,1.2,(0,200,0),2)
            cv.putText(frame,name,(x,y-50),2,1.2,(0,200,0),2)
        pixmap = convert_opencv_to_pixmap(frame)
        # 在QLabel上显示
        ui.label_2.setPixmap(pixmap)
        cv.waitKey(1) & 0xFF
    
        
#将OpenCV的BGR图像转换为PySide6的QPixmap
def convert_opencv_to_pixmap(cv_img):   
    # 转换颜色格式：BGR → RGB
    rgb_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    # 获取图像尺寸和通道数
    height, width, channel = rgb_img.shape
    bytes_per_line = width * 3  # 每个像素3个字节（RGB）
    # 创建QImage对象
    qimage = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    # 转换为QPixmap
    return QPixmap.fromImage(qimage)      


def recode():
    
    print("记录按钮被点击了!")
    creat()
   
    id=ui.id.text()
    name=ui.lineEdit.text()
    print(id)
    print(name)
    insertSql(id,name)
    capture()
    

def open():
    # ui.groupBox_2.hide()
    # ui.home.show()
   
    # now = datetime.now()
    # formatime = now.strftime("%Y-%m-%d %H:%M:%S")
    # ui.timela.setText(formatime)
    update_time()
    ui.openrecode.show()
    ui.timela.show()
    print("打卡按钮被点击了!")
    faceTest()

def pushButton_2():
    print("采集信息")
    save_Image()

def pushButton():
    train()
    print("训练成功")
    
def exit():
    sys.exit(app.exec())
    cap.release()
    
    print("退出")

def back():
       
    ui.groupBox.show()
    ui.openrecode.hide()
    ui.timela.hide()
    # ui.home.hide()
    # ui.groupBox_2.show()
    ui.plainTextEdit.clear()
    print("返回")  

def openbb():
     
    recongizer=cv.face.LBPHFaceRecognizer.create()
    recongizer.read("./trains/train.yml") 
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    local=face.detectMultiScale(frame)
    for x,y,w,h in local:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi=gray[y:y+h,x:x+w]
        id ,conf=recongizer.predict(roi)
        label=f"ID:{id} conf:{conf:.0f}%"     
    pixmap = convert_opencv_to_pixmap(roi)
        # 在QLabel上显示
    ui.label_4.setPixmap(pixmap)
    id=ui.id.text()
    name=ui.lineEdit.text()
    now = datetime.now()
    formatime = now.strftime("%Y-%m-%d %H:%M:%S")
   
    ui.plainTextEdit.appendPlainText(f"{id}:{name}在{formatime}打卡成功")
    print("打卡成功")
      
 
def setup_window():
    global app, window, ui
    
    app = QApplication(sys.argv)
    window = QMainWindow()
    
    # 创建UI对象
    ui = Ui_MainWindow()
    ui.setupUi(window)
    
    # 连接按钮信号和槽函数
    ui.save.clicked.connect(recode)
    ui.recode.clicked.connect(open)
    ui.pushButton_2.clicked.connect(pushButton_2)
    ui.pushButton.clicked.connect(pushButton)
    ui.exit.clicked.connect(exit)
    ui.back.clicked.connect(back)
    ui.openbb.clicked.connect(openbb)
    ui.openrecode.hide()
    ui.timela.hide()
    ui.home.hide()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
   
    
    setup_window()