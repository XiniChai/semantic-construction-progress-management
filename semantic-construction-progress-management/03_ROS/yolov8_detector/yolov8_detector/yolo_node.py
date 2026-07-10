# yolo_node.py
import rclpy, time, json, os
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        # parameters (declare + defaults)
        self.declare_parameter('model_path', '/home/xinichai/yolo_infer_env/models/best_v2.pt')
        self.declare_parameter('use_image_topic', False)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('test_image', '/home/xinichai/yolo_infer_env/lib/python3.10/site-packages/ultralytics/assets/step_18.jpg')
        self.declare_parameter('conf_thres', 0.35)
        # preprocessing params (train: 960x544, we will resize->960x540 then pad top/bottom 2)
        self.declare_parameter('proc_width', 960)
        self.declare_parameter('proc_height', 544)
        self.declare_parameter('resize_w', 960)
        self.declare_parameter('resize_h', 540)
        self.declare_parameter('pad_top', 2)
        self.declare_parameter('pad_bottom', 2)

        # load params
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.use_topic = self.get_parameter('use_image_topic').get_parameter_value().bool_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.test_image = self.get_parameter('test_image').get_parameter_value().string_value
        self.conf_thres = self.get_parameter('conf_thres').get_parameter_value().double_value
        self.resize_w = self.get_parameter('resize_w').get_parameter_value().integer_value
        self.resize_h = self.get_parameter('resize_h').get_parameter_value().integer_value
        self.pad_top = self.get_parameter('pad_top').get_parameter_value().integer_value
        self.pad_bottom = self.get_parameter('pad_bottom').get_parameter_value().integer_value
        self.proc_w = self.get_parameter('proc_width').get_parameter_value().integer_value
        self.proc_h = self.get_parameter('proc_height').get_parameter_value().integer_value

        # publisher
        self.pub = self.create_publisher(String, 'yolo/detections', 10)

        # bridge
        self.bridge = CvBridge()

        # load model
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found: {model_path}")
            raise FileNotFoundError(model_path)
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded model {model_path}")

        # either subscribe to Image topic or use local test image on a timer
        if self.use_topic:
            self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
            self.get_logger().info(f"Subscribed to topic {self.image_topic}")
        else:
            self.timer = self.create_timer(1.0, self.timer_cb)  # 1 Hz test
            self.get_logger().info(f"Using test image {self.test_image}")

    def preprocess_image(self, img):
        # img is BGR ndarray
        # 1) resize to resize_w x resize_h
        resized = cv2.resize(img, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
        # 2) pad top/bottom
        padded = cv2.copyMakeBorder(resized, self.pad_top, self.pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        # final shape should be proc_h x proc_w
        return padded

    def image_cb(self, msg: Image):
        # callback when using camera topic
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return
        self._run_inference_and_publish(cv_img)

    def timer_cb(self):
        img = cv2.imread(self.test_image)
        if img is None:
            self.get_logger().error(f"Cannot read test image {self.test_image}")
            return
        self._run_inference_and_publish(img)

    def _run_inference_and_publish(self, img):
        t0 = time.time()
        proc = self.preprocess_image(img)
        # model.predict accepts numpy image
        results = self.model.predict(source=proc, conf=self.conf_thres, stream=False)
        # typically results is list; take first result
        detections = []
        if len(results) > 0:
            r = results[0]
            # safe extraction: convert to python lists
            try:
                xyxy = r.boxes.xyxy.tolist()   # list of [x1,y1,x2,y2]
                confs = r.boxes.conf.tolist()
                cls_ids = r.boxes.cls.tolist()
            except Exception:
                # fallback: try to iterate boxes
                xyxy, confs, cls_ids = [], [], []
                for b in r.boxes:
                    # b.xyxy might be tensor/array
                    try:
                        xy = b.xyxy[0].tolist()
                    except:
                        xy = list(b.xyxy)
                    xyxy.append(xy)
                    try:
                        confs.append(float(b.conf[0]))
                        cls_ids.append(int(b.cls[0]))
                    except:
                        confs.append(float(b.conf))
                        cls_ids.append(int(b.cls))
            # assume single-class model -> class name = "shell"
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(float, box)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cls_id = int(cls_ids[i]) if len(cls_ids) > i else 0
                conf = float(confs[i]) if len(confs) > i else 0.0
                detections.append({
                    "id": i,
                    "class": "shell",  # fixed as per your training
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                })
        out = {
            "frame_id": "frame_test",
            "timestamp": time.time(),
            "proc_size": [self.proc_w, self.proc_h],
            "detections": detections
        }
        msg = String()
        msg.data = json.dumps(out)
        self.pub.publish(msg)
        self.get_logger().info(f"Published {len(detections)} detections (elapsed {time.time()-t0:.2f}s)")


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()