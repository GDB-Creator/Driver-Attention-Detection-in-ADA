# driver_monitor_video.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os


class DriverMonitor:
    def __init__(self, model_path='yolov8n-pose.pt'):
        """初始化驾驶员监控系统"""
        self.model = YOLO(model_path)
        # 头部关键点索引（COCO格式）
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4

        # 状态参数
        self.looking_away_counter = 0
        self.ALERT_THRESHOLD = 30  # 连续偏离30帧触发报警

        # 统计参数
        self.total_frames = 0
        self.forward_frames = 0
        self.alert_count = 0

    def calculate_head_pose(self, keypoints):
        """
        计算头部姿态角度
        基于鼻子和眼睛的位置关系估算头部偏转
        """
        try:
            nose = keypoints[self.NOSE]
            left_eye = keypoints[self.LEFT_EYE]
            right_eye = keypoints[self.RIGHT_EYE]

            # 计算面部中心点
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            face_center_y = (left_eye[1] + right_eye[1]) / 2

            # 计算头部偏转角度（简化算法）
            # 鼻子相对于面部中心的偏移量表示转头角度
            nose_offset_x = nose[0] - face_center_x
            nose_offset_y = nose[1] - face_center_y

            # 转换为角度（经验阈值）
            yaw_angle = nose_offset_x / 5  # 左右转头
            pitch_angle = nose_offset_y / 5  # 点头抬头
            # print(yaw_angle, pitch_angle)

            return yaw_angle, pitch_angle

        except (IndexError, TypeError):
            return None, None

    def is_looking_forward(self, yaw_angle, pitch_angle):
        """
        判断是否目视前方
        阈值根据实际场景调整
        """
        # 左右转头阈值 ±15度
        yaw_threshold = 15
        # 点头抬头阈值 ±12度
        pitch_threshold = 12

        if yaw_angle is None or pitch_angle is None:
            return True  # 无法检测时默认安全

        if abs(yaw_angle) < yaw_threshold and abs(pitch_angle) < pitch_threshold:
            return True
        return False

    def draw_head_direction(self, frame, keypoints, is_forward):
        """
        在图像上绘制头部方向指示
        """
        h, w = frame.shape[:2]

        # 获取鼻子位置作为参考点
        if keypoints is not None and len(keypoints) > self.NOSE:
            nose_x, nose_y = int(keypoints[self.NOSE][0]), int(keypoints[self.NOSE][1])

            # 绘制面部特征点
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

            # 绘制视线方向指示线
            if is_forward:
                # 向前看：绿色垂直线
                cv2.arrowedLine(frame, (nose_x, nose_y),
                                (nose_x, nose_y - 50), (0, 255, 0), 2)
                status_text = "Looking Forward"
                color = (0, 255, 0)
            else:
                # 偏离：红色指示线
                cv2.arrowedLine(frame, (nose_x, nose_y),
                                (nose_x + 30, nose_y - 20), (0, 0, 255), 2)
                status_text = "Looking Away"
                color = (0, 0, 255)

            # 显示状态文字
            cv2.putText(frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

    def process_frame(self, frame):
        """
        处理单帧图像
        """
        # 执行YOLOv8-Pose推理
        results = self.model(frame, conf=0.5, verbose=False)[0]

        # 初始化检测结果
        detection_result = {
            'has_driver': False,
            'looking_forward': True,
            'keypoints': None,
            'alert': False,
            'yaw': None,
            'pitch': None
        }

        # 处理检测结果
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            detection_result['has_driver'] = True

            # 获取第一个人的关键点
            keypoints = results.keypoints.data[0].cpu().numpy()
            print('guanjian:', keypoints)
            detection_result['keypoints'] = keypoints


            # 计算头部姿态
            yaw, pitch = self.calculate_head_pose(keypoints)
            detection_result['yaw'] = yaw
            detection_result['pitch'] = pitch

            # 判断是否目视前方
            is_forward = self.is_looking_forward(yaw, pitch)
            detection_result['looking_forward'] = is_forward

            # 更新统计
            if is_forward:
                self.forward_frames += 1

            # 更新偏离计数器
            if not is_forward:
                self.looking_away_counter += 1
                if self.looking_away_counter >= self.ALERT_THRESHOLD:
                    detection_result['alert'] = True
                    self.alert_count += 1
            else:
                self.looking_away_counter = max(0, self.looking_away_counter - 2)

            # 可视化
            frame = self.draw_head_direction(frame, keypoints, is_forward)

        self.total_frames += 1
        return frame, detection_result

    def process_stream(self, source, output_path=None, show_preview=True):
        """
        统一处理：本地视频 / RTSP流 / 摄像头
        """
        # 打开视频流
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"错误：无法打开视频源 -> {source}")
            return

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频源信息:")
        print(f"  - 分辨率: {width}x{height}")
        print(f"  - 帧率: {fps} FPS")
        if total_frames > 0:
            print(f"  - 总帧数: {total_frames}")
        else:
            print(f"  - 类型: 实时流(RTSP/摄像头)")
        print("\n开始处理... 按 'q' 退出")

        # 初始化视频写入器
        video_writer = None
        if output_path and width > 0 and height > 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流已断开或结束")
                break

            # 处理帧
            processed_frame, result = self.process_frame(frame)
            frame_count += 1

            # 保存视频
            if video_writer:
                video_writer.write(processed_frame)

            # 实时预览
            if show_preview:
                info_frame = processed_frame.copy()
                overlay = info_frame.copy()
                cv2.rectangle(overlay, (5, 5), (320, 160), (0, 0, 0), -1)
                info_frame = cv2.addWeighted(overlay, 0.6, info_frame, 0.4, 0)

                cv2.putText(info_frame, f"Status: {'Forward' if result['looking_forward'] else 'Away'}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if result['yaw'] is not None:
                    cv2.putText(info_frame, f"Yaw: {result['yaw']:.1f} deg",
                                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(info_frame, f"Pitch: {result['pitch']:.1f} deg",
                                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.putText(info_frame, f"Frame: {frame_count}",
                            (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if result['alert']:
                    cv2.putText(info_frame, "WARNING: DISTRACTED!",
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Driver Monitor", info_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 释放
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n处理结束 | 总帧数：{frame_count} | 耗时：{elapsed:.1f}s")

    def run_webcam(self):
        self.process_stream(source=0, show_preview=True)


# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='驾驶员监控系统（支持RTSP/视频/摄像头）')
    parser.add_argument('--source', '-s', type=str, help='视频源：本地路径 或 RTSP地址')
    parser.add_argument('--output', '-o', type=str, help='输出视频路径')
    parser.add_argument('--camera', '-c', action='store_true', help='直接使用摄像头')
    parser.add_argument('--model', '-m', default='yolov8n-pose.pt', help='模型路径')

    args = parser.parse_args()
    monitor = DriverMonitor(args.model)

    if args.camera:
        monitor.run_webcam()
    elif args.source:
        monitor.process_stream(source=args.source, output_path=args.output)
    else:
        print("使用示例：")
        print("  摄像头：python driver.py --camera")
        print("  本地视频：python driver.py --source test.mp4")
        print("  RTSP流：python driver.py --source rtsp://xxx.xxx.xxx.xxx:554/stream")