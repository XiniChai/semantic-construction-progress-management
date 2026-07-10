#!/usr/bin/env python3
#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from ultralytics import YOLO
import cv2, time, os, json
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import XSD
from datetime import datetime, timedelta, timezone
from std_msgs.msg import String
from typing import List

# ===== IoU计算函数 =====
def iou_xyxy(box1, box2):
    """计算两个xyxy框的IoU"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    a1 = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    a2 = max(0.0, (x4 - x3)) * max(0.0, (y4 - y3))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class StepFilterVideo(Node):
    """
    step_filter_node_video

    功能摘要：
    - 使用 YOLO 在视频帧中检测框
    - 与 metadata (step_view_index.json) 中的 bbox 计算 IoU，挑选最佳 step / bbox
    - 维护每个构件的首次检测时间 (hasActualStart) 与最近检测时间 (hasActualFinish)
    - 计算持续检测时长，只有当持续时长 >= install_min_duration（秒）才把状态标为 Installed，
      否则状态为 Detected（临时）
    - 周期性（save_interval 秒）写入 components_state.json 和 components_state.ttl，并发布 /state/json
    """
    def __init__(self):
        super().__init__('step_filter_node_video')

        # 参数声明（可通过 launch 覆盖）
        self.declare_parameter('video_path', '/home/xinichai/yolo_infer_env/camera_02_sample.mp4')
        self.declare_parameter('save_interval', 30)      # 写文件间隔（秒）
        self.declare_parameter('save_video', True)  #False
        self.declare_parameter('output_video', '/home/xinichai/yolo_infer_env/detection/camera_02_sample_results.mp4')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('log_interval', 30)
        self.declare_parameter('match_thresh', 0.3)
        self.declare_parameter('metadata_path', '/home/xinichai/metadata/step_view_index.json')
        self.declare_parameter('semantic_base_dir', '/home/xinichai/semantic_base')
        self.declare_parameter('install_min_duration', 30)  # 秒：持续检测达到该值才置 Installed,300

        # 读取参数
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.save_interval = int(self.get_parameter('save_interval').get_parameter_value().integer_value)
        self.save_video_flag = self.get_parameter('save_video').get_parameter_value().bool_value
        self.output_video_path = self.get_parameter('output_video').get_parameter_value().string_value
        self.conf_thres = float(self.get_parameter('conf_thres').get_parameter_value().double_value)
        self.log_interval = int(self.get_parameter('log_interval').get_parameter_value().integer_value)
        self.match_thresh = float(self.get_parameter('match_thresh').get_parameter_value().double_value)
        self.metadata_path = self.get_parameter('metadata_path').get_parameter_value().string_value
        self.semantic_base_dir = self.get_parameter('semantic_base_dir').get_parameter_value().string_value
        self.install_min_duration = int(self.get_parameter('install_min_duration').get_parameter_value().integer_value)

        # 确保语义目录存在，并设置 json/ttl 路径
        os.makedirs(self.semantic_base_dir, exist_ok=True)
        self.json_path = os.path.join(self.semantic_base_dir, 'components_state.json')
        self.ttl_path = os.path.join(self.semantic_base_dir, 'components_state.ttl')

        # 模型与元数据
        self.model_path = '/home/xinichai/yolo_infer_env/models/best_v2.pt'
        self.model = YOLO(self.model_path)
        with open(self.metadata_path, 'r') as f:
            self.ref_frames = json.load(f)

        # RDF 图（用于本地累积写入 components_state.ttl）
        self.g = Graph()
        if os.path.exists(self.ttl_path):
            try:
                self.g.parse(self.ttl_path, format='turtle')
            except Exception:
                self.g = Graph()
        self.CPMB = Namespace("http://www.semanticweb.org/cmpb/")
        self.g.bind('cpmb', self.CPMB)

        # Publisher: 发布 state/json 供 rdf_node 使用
        self.pub_state = self.create_publisher(String, 'state/json', 10)

        # 时间控制
        self.last_save_time = time.time()
        self.last_log_time = 0.0

        # 累积状态字典: cid -> info
        # info 包含: hasState, hasActualStart, hasActualFinish, hasActualDuration, conf
        self.accumulated = {}

        # 专门存 start/finish 时间用于不被覆盖
        # 使用本机带时区 ISO 字符串
        self.process_start_time = {}   # process_id -> ISO str
        self.process_finish_time = {}  # process_id -> ISO str

        # 打开视频
        self.cap = cv2.VideoCapture(self.video_path)
        # ===== 视频对应施工时间 =====
        self.video_start_time = datetime(
            2021, 8, 27,
            16, 00, 00
        )
        
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开视频: {self.video_path}")
            return

        # 可选保存输出视频
        self.writer = None
        if self.save_video_flag and self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
            self.writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
            self.get_logger().info(f"保存检测视频到: {self.output_video_path}")

        self.get_logger().info(f"StepFilterVideo started on {self.video_path}")
        # timer 每帧调用（大约）
        self.timer = self.create_timer(0.05, self.process_frame)


    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("视频播放完毕。")
            if self.writer:
                self.writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            return

        # YOLO 推理（直接在原帧上推理以保持显示分辨率一致）
        results = self.model(frame, conf=self.conf_thres, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy().tolist()
        pred_boxes = [box for box in detections if len(box) >= 4]
        # note: some ultralytics versions include score/class in boxes; we take first 4 values

        # 查找最佳匹配框及其IoU与Step（match_step 不写入 accumulated）
        best_process, best_iou, best_box, matched_ids = self.match_step(pred_boxes)

        # 可视化：仅显示满足阈值的最大 IoU 框（避免多框噪声）
        if best_box is not None and best_iou >= self.match_thresh:

            # ===== 定义初始时间 ====
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            if fps <= 0:
                fps = 30

            frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            elapsed_seconds = frame_id / fps

            actual_time = (
                self.video_start_time
                + timedelta(seconds=elapsed_seconds)
            )
            # ===== 定义初始时间 =====

            x1, y1, x2, y2 = map(int, best_box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"IoU={best_iou:.2f} | Process={best_process}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ===== 更新内存时间（不直接写文件），使用本地时区 ISO 字符串 =====
            t_iso = actual_time.strftime("%Y-%m-%dT%H:%M:%S")
            for cid in matched_ids:

                # ---------- Process级时间 ----------
                if best_process not in self.process_start_time:
                    self.process_start_time[best_process] = t_iso

                self.process_finish_time[best_process] = t_iso

                try:
                    start_dt = datetime.fromisoformat(self.process_start_time[best_process])
                    finish_dt = datetime.fromisoformat(self.process_finish_time[best_process])
                    duration_td = finish_dt - start_dt
                    duration_seconds = int(duration_td.total_seconds())
                    duration_str = str(duration_td)

                except Exception:
                    duration_seconds = 0
                    duration_str = "0:00:00"

                state = (
                    'Installed'
                    if duration_seconds >= self.install_min_duration
                    else 'Detected')

                self.accumulated[cid] = {
                    # 新增
                    'process_id': best_process,
                    # Component属性
                    'hasState': state,
                    # Process属性
                    'hasActualStart':
                        self.process_start_time[best_process],
                    'hasActualFinish':
                        self.process_finish_time[best_process],
                    'hasActualDuration':
                        duration_str,
                    'conf':
                        float(best_iou),
                    'duration_seconds':
                        duration_seconds
                }

        else:
            # 没有有效检测
            cv2.putText(frame, "No valid detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示或保存帧
        if self.save_video_flag and self.writer:
            self.writer.write(frame)
        else:
            cv2.imshow("YOLO Step Filter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("检测中止。")
                self.cap.release()
                cv2.destroyAllWindows()
                self.destroy_node()
                return

        now = time.time()
        # 每 save_interval 秒写一次 JSON/TTL 并发布 state/json
        if now - self.last_save_time >= self.save_interval:
            now_str = datetime.now().astimezone().isoformat()
            payload = {"timestamp": now_str, "state": self.accumulated}

            # 写 JSON（覆盖写入最新累积状态）
            try:
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.get_logger().error(f"写 JSON 失败: {e}")

            # 写 TTL（components_state.ttl），包含 accumulated 的三元组（覆盖写入）
            try:
                g_local = Graph()
                g_local.bind('cpmb', self.CPMB)
                for cid, info in self.accumulated.items():

                    comp_subj = URIRef(f"http://www.semanticweb.org/cmpb/{cid}")
                    process_id = info.get('process_id')
                    proc_subj = None
                    if process_id:
                        proc_subj = URIRef(f"http://www.semanticweb.org/cmpb/{process_id}")

                    if 'hasState' in info:
                        g_local.add((comp_subj, self.CPMB.hasState, Literal(info['hasState'], datatype=XSD.string)))
                    if proc_subj and 'hasActualStart' in info:
                        g_local.add((proc_subj, self.CPMB.hasActualStart, Literal(info['hasActualStart'], datatype=XSD.dateTime)))
                    if proc_subj and 'hasActualFinish' in info:
                        g_local.add((proc_subj, self.CPMB.hasActualFinish, Literal(info['hasActualFinish'], datatype=XSD.dateTime)))
                    if proc_subj and 'hasActualDuration' in info:
                        g_local.add((proc_subj, self.CPMB.hasActualDuration, Literal(info['hasActualDuration'], datatype=XSD.string)))
                    
                    if proc_subj:
                        g_local.add((proc_subj, self.CPMB.hasObject, comp_subj))

                    if 'conf' in info:
                        try:
                            g_local.add((comp_subj, self.CPMB.confidence, Literal(float(info['conf']), datatype=XSD.float)))
                        except Exception:
                            g_local.add((comp_subj, self.CPMB.confidence, Literal(str(info.get('conf')))))
                    # 可选：记录持续秒数为辅助属性（非必需）
                    if proc_subj and 'duration_seconds' in info:
                        try:
                            g_local.add((proc_subj, self.CPMB.durationSeconds, Literal(int(info['duration_seconds']))))
                        except Exception:
                            pass
                g_local.serialize(destination=self.ttl_path, format='turtle')
            except Exception as e:
                self.get_logger().error(f"写 TTL 失败: {e}")

            # 发布 state/json 话题（供 rdf_node 处理并写入 base graph）
            try:
                msg = String()
                msg.data = json.dumps(payload, ensure_ascii=False)
                self.pub_state.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Publish state/json failed: {e}")

            self.get_logger().info(f"{self.json_path}, {self.ttl_path} ")  #✅ 状态已保存与发布：({now_str})
            self.last_save_time = now

        # 定时日志输出（只输出简要信息）
        if now - self.last_log_time > self.log_interval:
            if best_iou >= self.match_thresh:
                self.get_logger().info(f"Best IoU={best_iou:.2f}, Process={best_process}") #[{time.strftime('%H:%M:%S')}] Best IoU={best_iou:.2f}, Process={best_process}
            else:
                self.get_logger().info(f"No detection above threshold ({self.match_thresh})") #[{time.strftime('%H:%M:%S')}] No detection above threshold ({self.match_thresh})
            self.last_log_time = now


    def match_step(self, pred_boxes: List[List[float]]):
        """
        匹配逻辑：返回 (best_process, best_iou, best_box, matched_ids)
        注意：此函数**不**负责写入 accumulated / ttl，只做匹配决策
        """
        best_process, best_iou, best_box, matched_ids = None, 0.0, None, []

        for frame_data in self.ref_frames:
            step = frame_data.get('step', -1)
            comps = frame_data.get('components', [])
            for pb in pred_boxes:
                # ensure pb first 4 numbers
                pb4 = pb[:4]
                for rc in comps:
                    bbox_ref = rc.get('bbox_2d')
                    if not bbox_ref or len(bbox_ref) != 4:
                        continue
                    iou = iou_xyxy(pb4, bbox_ref)
                    if iou > best_iou:
                        best_iou, best_process, best_box = iou, step, pb4
                        matched_ids = [rc.get('id')]

        return best_process, best_iou, best_box, matched_ids


    def save_state(self):
        """可以被外部调用：强制写一次当前 accumulated 到 json/ttl 并 publish"""
        now_str = datetime.now().astimezone().isoformat()
        payload = {"timestamp": now_str, "state": self.accumulated}
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"手动写 JSON 失败: {e}")

        try:
            g_local = Graph()
            g_local.bind('cpmb', self.CPMB)
            for cid, info in self.accumulated.items():
                comp_subj = URIRef(f"http://www.semanticweb.org/cmpb/{cid}")
                
                process_id = info.get('process_id')
                proc_subj = None

                if process_id:
                    proc_subj = URIRef(f"http://www.semanticweb.org/cmpb/{process_id}")

                if 'hasState' in info:
                    g_local.add((comp_subj, self.CPMB.hasState, Literal(info['hasState'], datatype=XSD.string)))
                if proc_subj and 'hasActualStart' in info:
                    g_local.add((proc_subj, self.CPMB.hasActualStart, Literal(info['hasActualStart'], datatype=XSD.dateTime)))
                if proc_subj and 'hasActualFinish' in info:
                    g_local.add((proc_subj, self.CPMB.hasActualFinish, Literal(info['hasActualFinish'], datatype=XSD.dateTime)))
                if proc_subj and 'hasActualDuration' in info:
                    g_local.add((proc_subj, self.CPMB.hasActualDuration, Literal(info['hasActualDuration'], datatype=XSD.string)))
                if proc_subj:
                    g_local.add((proc_subj, self.CPMB.hasObject, comp_subj))

                if 'conf' in info:
                    try:
                        g_local.add((comp_subj, self.CPMB.confidence, Literal(float(info['conf']), datatype=XSD.float)))
                    except Exception:
                        g_local.add((comp_subj, self.CPMB.confidence, Literal(str(info.get('conf')))))
            g_local.serialize(destination=self.ttl_path, format='turtle')
        except Exception as e:
            self.get_logger().error(f"手动写 TTL 失败: {e}")

        # publish
        try:
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_state.publish(msg)
        except Exception as e:
            self.get_logger().error(f"手动 Publish state/json failed: {e}")

        self.get_logger().info(f"手动保存：{self.json_path}, {self.ttl_path} ({now_str})")
        self.last_save_time = time.time()


# ===== 主函数 =====
def main(args=None):
    rclpy.init(args=args)
    node = StepFilterVideo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

