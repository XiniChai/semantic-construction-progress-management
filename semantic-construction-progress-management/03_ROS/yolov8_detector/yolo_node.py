# yolo_node.py
import rclpy, time, json
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.pub = self.create_publisher(String, 'yolo/detections', 10)
        self.timer = self.create_timer(0.5, self.timer_cb)  # 2 Hz
        self.bridge = CvBridge()

        # 指定模型路径（修改为你的路径）
        model_path = "/home/xinichai/yolo_infer_env/models/yolov8n.pt"
        self.model = YOLO(model_path)

        # 测试用图片（也可接摄像头）
        self.test_image = "/home/xinichai/yolo_infer_env/lib/python3.10/site-packages/ultralytics/assets/bus.jpg"
        self.get_logger().info("YOLO node ready")

    def timer_cb(self):
        t0 = time.time()
        # 加载图像（你也可以从 /camera/image_raw 订阅）
        img = cv2.imread(self.test_image)
        if img is None:
            self.get_logger().error(f"Cannot read {self.test_image}")
            return

        results = self.model.predict(source=img, conf=0.35, stream=False)
        # results 可能有多个 frames (这里仅1)
        out = {"frame_id": "test", "timestamp": time.time(), "detections": []}
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                # box.xyxy, box.conf, box.cls
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy)
                cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                name = self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)
                out["detections"].append({
                    "id": i,
                    "class": name,
                    "conf": conf,
                    "bbox": xyxy
                })
        msg = String()
        msg.data = json.dumps(out)
        self.pub.publish(msg)
        self.get_logger().info(f"Published {len(out['detections'])} detections, elapsed {time.time()-t0:.2f}s")


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
