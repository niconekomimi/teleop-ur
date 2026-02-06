import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthViewer(Node):
    def __init__(self):
        super().__init__('depth_viewer')
        self.bridge = CvBridge()
        # 订阅对齐后的深度图
        self.sub = self.create_subscription(
            Image, 
            '/camera/camera/aligned_depth_to_color/image_raw', 
            self.callback, 
            10)

    def callback(self, msg):
        try:
            # 1. 拿到 16位 原始深度图 (单位: 毫米)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            
            # 2. 把它转换成可视化的 8位 彩色图
            # 这里的 alpha=0.01 是为了把近距离的对比度拉大，
            # 让 0-1米 范围内的颜色变化更明显
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.4), 
                cv2.COLORMAP_JET
            )

            # 3. 显示
            cv2.imshow("Filtered Depth Stream", depth_colormap)
            cv2.waitKey(1)
        except Exception as e:
            print(e)

def main():
    rclpy.init()
    node = DepthViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()