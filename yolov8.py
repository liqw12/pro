import sys
import typing
if not hasattr(typing, 'Self'):
     typing.Self = type('Self', (), {})
from PySide6.QtWidgets import QApplication, QMainWindow
from cars import Ui_closevideo  # 导入生成的UI类
import cv2 as cv
from ultralytics import YOLO
from typing_extensions import Self

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QFileDialog
)
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from PIL import Image
import numpy as np 
import sqlite3
import time
from datetime import datetime
from PySide6.QtCore import QTimer
from PIL import ImageDraw,ImageFont,Image

class MyClass:
    def method(self) -> Self:
        return self     
# 创建全局变量
app = None
window = None
ui = None
grade=["优秀","良好","较差","需要疏通"]
cap = cv.VideoCapture(0)
color=(0,255,0)

carNum=0
vanNum=0
truckNum=0
busNum=0
tracked_vehicles = {}
next_vehicle_id = 0  
total=0
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

#图片检测
def imgTest(path):
    id=0
    global carNum, vanNum, truckNum, busNum,total
    img=cv.imread(path)
    model = YOLO("car.pt")
    results = model(path)

    for result in results[0].boxes:
        classid =result.cpu().cls.squeeze().item()
        conf=result.cpu().conf.squeeze().item()
        x1,y1,x2,y2=result.cpu().xyxy.squeeze().numpy().astype(int)
        label=f"{model.names[classid]} conf:{conf:.2}"
        if model.names[classid]=='car' and conf > 0.5:
                carNum+=1
        if  model.names[classid]=='bus' and conf > 0.5:
            busNum+=1 
        if  model.names[classid]=='truck'and conf > 0.5:
            truckNum+=1
        if  model.names[classid]=='van' and conf > 0.5:
            vanNum+=1    
        total+=1
        cv.putText(img,label,(x1,y1-10),cv.FONT_HERSHEY_COMPLEX,1.2,color,2)
        cv.rectangle(img,(x1,y1),(x2,y2),color,2)
        ui.xmax.setText(str(x1)) 
        ui.xmin.setText(str(x2))
        ui.ymax.setText(str(y1))
        ui.ymin.setText(str(y2) )  
        ui.van.setText(str(vanNum))
        ui.truck.setText(str(truckNum))
        ui.car.setText(str(carNum))
        ui.bus.setText(str(busNum))
        ui.conf.setText(f"{conf:.2}")  
        text=f"id:{id}  分类:{model.names[classid]} conf:{conf:.2} 位置:{int(x1),int(x2),int(y1),int(y2)}"
        ui.plainTextEdit.appendPlainText(text)
        id+=1
    if total<100:    
        ui.grade.setText("良好")
    ui.plainTextEdit.appendPlainText("")    
    pixmap = convert_opencv_to_pixmap(img)
        # 在QLabel上显示
    ui.video.setPixmap( pixmap)

 # 视频检测
#视频检测
def videoTest2(path):
    list=[]
    global carNum, vanNum, truckNum, busNum,total  # 声明使用全局变量
    list=[]
    color = (255, 0, 0)  # BGR格式的蓝色

    # 1. 加载视频
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("无法打开视频文件:", path)
        return
    
    # 2. 加载YOLO模型
    model = YOLO("car.pt")  # 确保 "car.pt" 是正确的模型路径
    
    # 3. 初始化计数器和跟踪变量
    carNum = 0
    vanNum = 0
    truckNum = 0
    busNum = 0
    
    # 用于跟踪已计数的车辆
    # 字典结构: {vehicle_id: {'class': 类别名称, 'box': [x1,y1,x2,y2], 'conf': 置信度}}
    tracked_vehicles = {}
    next_vehicle_id = 1  # 从1开始编号
    
    # 用于记录所有检测到的车辆信息（可选，如果需要在UI上显示详细信息）
    detected_vehicles_info = []  # 列表存储字典，每个字典包含车辆信息
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束
        
        # 4. 使用模型进行检测
        results = model(frame)
        
        # 5. 提取检测结果
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # [N,4] 格式 [x1,y1,x2,y2]
        confs = results[0].boxes.conf.cpu().numpy()              # [N,] 置信度
        class_ids = results[0].boxes.cls.cpu().numpy()           # [N,] 类别 ID
        
        # 6. 单帧内NMS去重（虽然YOLOv8已经做了NMS，但可以再做一次更严格的）
        nms_indices = cv.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), score_threshold=0.5, nms_threshold=0.4)
        if isinstance(nms_indices, np.ndarray):
            nms_indices = nms_indices.flatten().tolist()  # 确保 nms_indices 是列表
        else:
            nms_indices = []  # 如果没有检测到，可能返回非列表
        
        filtered_boxes = [boxes[i] for i in nms_indices] if nms_indices else []
        filtered_confs = [confs[i] for i in nms_indices] if nms_indices else []
        filtered_class_ids = [class_ids[i] for i in nms_indices] if nms_indices else []
        
        # 7. 多帧间关联 & 计数
        current_frame_vehicles = []  # 记录当前帧已分配ID的车辆 (id, box, cls, conf)
        
        for box, conf, cls_id in zip(filtered_boxes, filtered_confs, filtered_class_ids):
            cls_name = model.names[int(cls_id)]
            
            # 只处理我们要计数的车辆类型
            if cls_name not in ['car', 'bus', 'truck', 'van']:
                continue
            
            matched = False
            # 检查是否与已计数的车辆匹配（IoU > 0.5）
            for vid, vehicle_info in tracked_vehicles.items():
                prev_box = vehicle_info['box']
                iou = calculate_iou(box, prev_box)
                if iou > 0.5:  # IoU阈值
                    matched = True
                    # 更新现有车辆的信息（如果需要）
                    # 这里选择不更新编号，只更新位置和置信度（可选）
                    vehicle_info['box'] = box
                    vehicle_info['conf'] = conf
                    current_frame_vehicles.append((vid, box, cls_name, conf))
                    break


            if not matched:
                # 新车辆，分配新ID并计数
                if cls_name == 'car' and conf > 0.5:
                    carNum += 1
                elif cls_name == 'bus' and conf > 0.5:
                    busNum += 1
                elif cls_name == 'truck' and conf > 0.5:
                    truckNum += 1
                elif cls_name == 'van' and conf > 0.5:
                    vanNum += 1
                total+=1
                # 分配新ID
                vehicle_id = next_vehicle_id
                next_vehicle_id += 1
                
                # 记录车辆信息
                tracked_vehicles[vehicle_id] = {
                    'class': cls_name,
                    'box': box,
                    'conf': conf
                }
                
                current_frame_vehicles.append((vehicle_id, box, cls_name, conf))
                
                # 可选：记录到 detected_vehicles_info 中（如果需要在UI上显示详细信息）
                detected_vehicles_info.append({
                    'id': vehicle_id,
                    'class': cls_name,
                    'box': box,
                    'conf': conf
                })
        
        for vid, box, cls_name, conf in current_frame_vehicles:
            x1, y1, x2, y2 = box
            label = f"ID:{vid} {cls_name} conf:{conf:.2f}"
            
            
            if vid not in list:
                text=f"id:{vid}  分类:{cls_name} conf:{conf:.2} 位置:{int(x1),int(x2),int(y1),int(y2)}"
                ui.plainTextEdit.appendPlainText(text)
                list.append(vid)
            # 绘制边界框
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, color, 2)
      
        if len(current_frame_vehicles) > 0:
            first_box = current_frame_vehicles[0][1]
            ui.xmax.setText(str(first_box[2]))  # xmax
            ui.xmin.setText(str(first_box[0]))  # xmin
            ui.ymax.setText(str(first_box[3]))  # ymax
            ui.ymin.setText(str(first_box[1]))  # ymin
            ui.van.setText(str(vanNum))
            ui.truck.setText(str(truckNum))
            ui.car.setText(str(carNum))
            ui.bus.setText(str(busNum))
            ui.conf.setText(f"{conf:.2}")   
        if total>20:
            ui.grade.setText("较差")
        # 11. 转换并显示图像
        pixmap = convert_opencv_to_pixmap(frame)
        ui.video.setPixmap(pixmap)
        
        # 11. 退出条件
        cd = cv.waitKey(1)
        if cd == 27:  # ESC键退出
            break
    
    # 12. 释放资源
    cap.release()
    cv.destroyAllWindows()


   

# 计算IoU的辅助函数
def calculate_iou(box1, box2):
    # box格式: [x1,y1,x2,y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    # 避免除以0
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# def browse_image():
#     """浏览并选择图像文件"""
#     file_path, _ = QFileDialog.getOpenFileName(
#         # self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)"
#     )
    print(file_path)
def videoTest(path):
    global carNum, vanNum, truckNum, busNum  # 声明使用全局变量
    
    
    color=(255,0,0)
    #1
    cap=cv.VideoCapture(path)


    model=YOLO("car.pt")
    #2
    while True:
        ret,frame=cap.read()
        results=model(frame)
        print(results[0].boxes)


        for result in results[0].boxes:
            classid =result.cpu().cls.squeeze().item()
            conf=result.cpu().conf.squeeze().item()
            x1,y1,x2,y2=result.cpu().xyxy.squeeze().numpy().astype(int)
            label=f"{model.names[classid]} conf:{conf:.2}"
            print(model.names[classid])
            if model.names[classid]=='car' and conf > 0.5:
                carNum+=1
            if  model.names[classid]=='bus' and conf > 0.5:
                busNum+=1 
            if  model.names[classid]=='truck'and conf > 0.5:
                truckNum+=1
            if  model.names[classid]=='van' and conf > 0.5:
                vanNum+=1           
                # cn_label = chinese_name.get(model.names[classid],model.names[classid])  # 如果找不到英文名，保留原英文名
                # # 1. 将 OpenCV 的 BGR 图像转换为 PIL 的 RGB 格式
                # pil_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                # draw = ImageDraw.Draw(pil_image)
                
                # # 2. 设置中文字体（确保 "simsun.ttc" 字体文件存在）
                # try:
                #     font = ImageFont.truetype("simsun.ttc", 20, encoding="utf-8")
                # except IOError:
                #     print("警告：未找到中文字体文件 'simsun.ttc'，将使用默认字体（可能不支持中文）")
                #     font = ImageFont.load_default()

                # # 3. 在 PIL 图像上绘制中文标签
                # draw.text((x1, y1 - 20), cn_label, fill=(255, 0, 0), font=font)
                
                # # 4. 将 PIL 图像转换回 OpenCV 的 BGR 格式
                # frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
            ui.xmax.setText(str(x1)) 
            ui.xmin.setText(str(x2))
            ui.ymax.setText(str(y1))
            ui.ymin.setText(str(y2) )  
            ui.van.setText(str(vanNum))
            ui.truck.setText(str(truckNum))
            ui.car.setText(str(carNum))
            ui.bus.setText(str(busNum))
            cv.putText(frame,label,(x1,y1-10),cv.FONT_HERSHEY_COMPLEX,1.2,color,2)
            cv.rectangle(frame,(x1,y1),(x2,y2),color,2)


        pixmap = convert_opencv_to_pixmap(frame)
            # 在QLabel上显示
        ui.video.setPixmap( pixmap)
        cd=cv.waitKey(1)
        if cd==27:
            break
# 打开图片按键   
def openpicture():
    ui.plainTextEdit.clear()
    file_path, _ = QFileDialog.getOpenFileName(None, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)")
    print(file_path)
    imgTest(file_path)


# 摄像头检测
def capOpen():
    list=[]
    global carNum, vanNum, truckNum, busNum,tootal  # 声明使用全局变量   
    color = (255, 0, 0)
    #摄像头
    model = YOLO("car.pt")
    carNum = 0
    vanNum = 0
    truckNum = 0
    busNum = 0
    total=0
    #用于跟踪已计数的车辆（ID: (x1,y1,x2,y2)）
    tracked_vehicles = {}
    next_vehicle_id = 0
    
    #开始视频处理循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用模型进行检测
        results = model(frame)
        
        #  提取检测结果
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # [N,4] 格式 [x1,y1,x2,y2]
        confs = results[0].boxes.conf.cpu().numpy()              # [N,] 置信度
        class_ids = results[0].boxes.cls.cpu().numpy()           # [N,] 类别 ID
        
        #  单帧内NMS去重（虽然YOLOv8已经做了NMS，但可以再做一次更严格的）
        nms_indices = cv.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), 0.5, 0.4)
        filtered_boxes = boxes[nms_indices]
        filtered_confs = confs[nms_indices]
        filtered_class_ids = class_ids[nms_indices]
        
        # 多帧间关联 & 计数
        current_frame_vehicles = []  # 记录当前帧已分配ID的车辆 (id, box, cls)
        
        for box, conf, cls_id in zip(filtered_boxes, filtered_confs, filtered_class_ids):
            cls_name = model.names[int(cls_id)]
            
            # 只处理我们要计数的车辆类型
            if cls_name not in ['car', 'bus', 'truck', 'van']:
                continue
                
            matched = False
            # 检查是否与已计数的车辆匹配（IoU > 0.5）
            for vid, (prev_box, _) in tracked_vehicles.items():
                iou = calculate_iou(box, prev_box)
                if iou > 0.5:  # IoU阈值
                    matched = True
                    current_frame_vehicles.append((vid, box, cls_name))
                    break
            
            if not matched:
                # 新车辆，分配新ID并计数
                if cls_name == 'car' and conf > 0.5:
                    carNum += 1
                elif cls_name == 'bus' and conf > 0.5:
                    busNum += 1
                elif cls_name == 'truck' and conf > 0.5:
                    truckNum += 1
                elif cls_name == 'van' and conf > 0.5:
                    vanNum += 1
                total+=1
                tracked_vehicles[next_vehicle_id] = (box, cls_name)
                current_frame_vehicles.append((next_vehicle_id, box, cls_name))
                next_vehicle_id += 1
        
        ui.van.setText(str(vanNum))
        ui.truck.setText(str(truckNum))
        ui.car.setText(str(carNum))
        ui.bus.setText(str(busNum))
        

        for vid, box, cls_name in current_frame_vehicles:
            x1, y1, x2, y2 = box
            conf = next((conf for b, conf in zip(filtered_boxes, filtered_confs) if all(b[i] == box[i] for i in range(4))), 0.0)
            label = f"{cls_name} conf:{conf:.2f}"
            
            if len(current_frame_vehicles) > 0:
                first_box = current_frame_vehicles[0][1]
                ui.xmax.setText(str(first_box[0]))
                ui.xmin.setText(str(first_box[2]))
                ui.ymax.setText(str(first_box[1]))
                ui.ymin.setText(str(first_box[3]))
                
            if vid not in list:
                text=f"id:{vid}  分类:{cls_name} conf:{conf:.2} 位置:{int(x1),int(x2),int(y1),int(y2)}"
                ui.plainTextEdit.appendPlainText(text)
                list.append(vid)
            if total<10:
                ui.grade.setText("优秀")
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, color, 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        pixmap = convert_opencv_to_pixmap(frame)
        ui.video.setPixmap(pixmap)
 
        cd = cv.waitKey(1)
        if cd == 27:  # ESC键退出
            break
    
    # 13. 释放资源
    cap.release()
    cv.destroyAllWindows()

def opencap():
    ui.plainTextEdit.clear()
    print("摄像头已打开")
    capOpen()
   
#打开视频
def openvideo():
    ui.plainTextEdit.clear()
    file_path, _ = QFileDialog.getOpenFileName(None, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.mp4)")
    # print("openvideo")
    videoTest2(file_path)
    print(file_path)

# 保存 按键
def save():
    print("save")
    file_path, _ = QFileDialog.getOpenFileName(None, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.mp4)")
    


# 关闭视频按键
def closecap():
    ui.plainTextEdit.clear()
    cap.release()
    cv.destroyAllWindows()
    print("close")

# 退出按键
def exit():
    print("exit")
    sys.exit(app.exec())

def setup_window():
    global app, window, ui
    
    app = QApplication(sys.argv)
    window = QMainWindow()
    
    # 创建UI对象
    ui = Ui_closevideo()
    ui.setupUi(window)
    
    # 连接按钮信号和槽函数
    ui.openpicture.clicked.connect(openpicture)
    ui.opencap.clicked.connect(opencap)
    ui.openvideo.clicked.connect(openvideo)
    ui.save.clicked.connect(save)
    ui.closecap.clicked.connect(closecap)
    ui.exit.clicked.connect(exit)
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    setup_window()