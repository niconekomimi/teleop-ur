import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import math

class GestureTestNode(Node):
    def __init__(self):
        super().__init__('gesture_test_node')
        
        # 订阅 RealSense 图像
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # <--- 改成这个
            self.listener_callback,
            10) # 如果还是不行，记得把这里改成 qos_profile_sensor_data
        
        self.bridge = CvBridge()
        
        # MediaPipe 初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # === 🔧 调试参数 ===
        # 捏合阈值：食指和大拇指指尖距离小于这个像素值，就算“抓紧”
        # 你可以根据实际体验调整这个数字 (建议 30-60 之间)
        self.PINCH_THRESHOLD = 40 
        
        self.get_logger().info("✊ 手势测试节点已启动！请对着相机做捏合动作...")

    def calculate_distance(self, p1, p2, w, h):
        """计算两点间的像素欧氏距离"""
        x1, y1 = p1.x * w, p1.y * h
        x2, y2 = p2.x * w, p2.y * h
        return math.hypot(x2 - x1, y2 - y1)

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = cv_image.shape
            
            # 转 RGB 喂给 MediaPipe
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            state_text = "Searching..."
            color = (200, 200, 200) # 灰色

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 1. 画骨架
                    self.mp_drawing.draw_landmarks(
                        cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 2. 获取关键点：大拇指(4) 和 食指(8)
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    
                    # 3. 计算距离
                    dist = self.calculate_distance(thumb_tip, index_tip, w, h)
                    
                    # 4. 判断状态
                    if dist < self.PINCH_THRESHOLD:
                        state_text = f"GRASP! (dist={int(dist)})"
                        color = (0, 255, 0) # 绿色代表抓取
                    else:
                        state_text = f"OPEN   (dist={int(dist)})"
                        color = (0, 0, 255) # 红色代表松开

                    # 5. 在手指间画一条线，直观看到距离
                    p1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    p2 = (int(index_tip.x * w), int(index_tip.y * h))
                    cv2.line(cv_image, p1, p2, color, 2)
                    cv2.circle(cv_image, ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2), 4, color, -1)

            # 在屏幕左上角显示状态
            cv2.putText(cv_image, state_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow("Gesture Test", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = GestureTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()