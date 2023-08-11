import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from ScreenCapture import ScreenCapture

frame_count = 1


def run(sc):
    global frame_count
    start_time = time.time()
    fps = 1
    # 实例化 YOLO 模型
    model = YOLO(r'../models/yolov8n.engine')
    # 设置字体
    fontStyle = ImageFont.truetype(r'../../resources/font/SIMSUN.ttf', size=48, encoding='utf-8')  # 中文字体文件
    # 循环体
    while True:
        # 截屏
        img = Image.fromarray(np.uint8(sc.grab_screen_mss()))
        # 实例化一个图像绘制对象
        draw = ImageDraw.Draw(img)
        # 利用模型进行图像预测
        results = model.predict(source=img, conf=0.75, iou=0.75)
        # 遍历结果，detection
        for result in results:
            # detection
            if len(result.boxes.xyxy) > 0:
                boxes_conf = np.array(result.boxes.conf.tolist())
                # 确定检测结果的数量
                boxes_conf_n = len(boxes_conf)
                if boxes_conf_n > 0:
                    boxes_xyxy = result.boxes.xyxy.tolist()
                    boxes_cls = result.boxes.cls.tolist()
                    boxes_xy_center = []
                    for i, box_xyxy in enumerate(boxes_xyxy):
                        draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                        draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                                  font=fontStyle)
                        boxes_xy_center.append((((box_xyxy[0] + box_xyxy[2]) / 2) + sc.capture_left,
                                                ((box_xyxy[1] + box_xyxy[3]) / 2) + sc.capture_top))
        # 计算并显示FPS
        time_diff = time.time() - start_time
        fps = 1 / time_diff
        draw.text((30, 40), f"FPS: {fps:.2f}", (0, 0, 255), font=fontStyle)
        cv2.namedWindow(sc.window_name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 根据窗口大小设置图片大小
        cv2.resizeWindow(sc.window_name, sc.display_window_width, sc.display_window_height)
        cv2.imshow(sc.window_name, np.array(img))
        if cv2.waitKey(1) & 0XFF == sc.exit_code:  # 默认：ESC
            cv2.destroyAllWindows()
            os._exit(0)


if __name__ == '__main__':
    sc = ScreenCapture(capture_region=(1, 1))
    run(sc)
