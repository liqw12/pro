# 导入模块
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
from mediapipe.framework.formats import landmark_pb2
import time
import math

box_size=100
box_x=100
box_y=100
last_pinch_pos = None  # 记录上一次捏合位置
color=(0,0,255)

# 初始化 Mediapipe 手势识别
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 创建手势识别器实例（VIDEO 模式）
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO  # 使用 VIDEO 模式
)
recognizer = GestureRecognizer.create_from_options(options)
# 初始化摄像头
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()  
    if not ret:
        print("无法读取摄像头画面！")
        break

    # 镜像翻转（可选）
    frame = cv.flip(frame, 1) 
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
    # 处理手部关键点
    if gesture_recognition_result.hand_landmarks:
        for hand_landmarks in gesture_recognition_result.hand_landmarks:
            # 创建 NormalizedLandmarkList 对象用于绘制
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])
            
            # 绘制手部关键点和连接线
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 获取拇指和食指的坐标（归一化坐标）
            # Mediapipe 手部关键点索引:
            # 4: 大拇指指尖
            # 8: 食指指尖
            thumb_tip = hand_landmarks[4]  # 大拇指指尖
            index_tip = hand_landmarks[8]  # 食指指尖
            
            # 打印归一化坐标
            print(f"拇指坐标 (归一化): x={thumb_tip.x:.4f}, y={thumb_tip.y:.4f}, z={thumb_tip.z:.4f}")
            print(f"食指坐标 (归一化): x={index_tip.x:.4f}, y={index_tip.y:.4f}, z={index_tip.z:.4f}")
            
            # 将归一化坐标转换为像素坐标
            height, width = frame.shape[:2]
            thumb_x = int(thumb_tip.x * width)  # 拇指 x 坐标（像素）
            thumb_y = int(thumb_tip.y * height)  # 拇指 y 坐标（像素）
            index_x = int(index_tip.x * width)  # 食指 x 坐标（像素）
            index_y = int(index_tip.y * height)  # 食指 y 坐标（像素）
            
            # 打印像素坐标
            print(f"拇指坐标 (像素): x={thumb_x}, y={thumb_y}")
            print(f"食指坐标 (像素): x={index_x}, y={index_y}")
            if  math.sqrt( (thumb_x - index_x )**2 + (thumb_y - index_y)**2)<40 :
                 # 检查是否在方框内
                if (box_x <= thumb_x <= box_x + 100 and 
                    box_y <= thumb_y <= box_y + 100):
                    print("捏合点在方框内!")
                    
                   # 更新方框位置到捏合点
                    box_x, box_y = thumb_x - box_size//2, thumb_y - box_size//2
                    last_pinch_pos = (thumb_x, thumb_y)
                    #color
                    color=(255,0,0,)
                elif last_pinch_pos:
                    # 如果之前在方框内，继续跟随移动
                    box_x, box_y = last_pinch_pos[0] - box_size//2, last_pinch_pos[1] - box_size//2 
    
        # 透明红色方块 (BGR格式)

    
    
  # 绘制方框（带透明度）
    overlay = frame.copy()
    cv.rectangle(overlay, (box_x, box_y), (box_x + box_size, box_y + box_size), color, -1)
    cv.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    color=(0,0,255)

    cv.imshow('image', frame)

    # 按ESC键退出
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()