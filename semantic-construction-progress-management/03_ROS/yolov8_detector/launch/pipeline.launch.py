from launch import LaunchDescription
from launch_ros.actions import Node
import datetime
import os

# 固定虚拟环境 Python 解释器路径
PYTHON_EXEC = "/home/xinichai/yolo_infer_env/bin/python"

# 自动生成唯一输出视频路径，防止覆盖
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = f"/home/xinichai/yolo_infer_env/detection/output_{timestamp}.avi"

def generate_launch_description():
    # 确保检测结果目录存在
    os.makedirs("/home/xinichai/yolo_infer_env/detection", exist_ok=True)
    os.makedirs("/home/xinichai/semantic_base", exist_ok=True)

    return LaunchDescription([

        # ====================================================
        # 🧩 Step + YOLO 检测节点：实时视频 + 构件识别 + 状态生成
        # ====================================================
        Node(
            package='yolov8_detector',
            executable='step_filter_node_video',
            output='screen',
            prefix=[PYTHON_EXEC, ' '],
            parameters=[{
                # 视频输入路径
                'video_path': '/home/xinichai/yolo_infer_env/camera_02_sample.mp4',

                # BIM 构件元数据（包含Step与View信息）
                'metadata_path': '/home/xinichai/metadata/step_view_index.json',

                # IoU 匹配阈值
                'match_thresh': 0.3,

                # 检测置信度阈值
                'conf_thres': 0.5,

                # 每隔多少秒写入状态文件（30秒）
                'save_interval': 30,

                # 日志打印间隔（秒）
                'log_interval': 30,

                # 输出视频路径（避免覆盖）
                'output_video': output_video_path,

                # RDF 基础语义图（用于语义写入）
                'base_rdf': '/home/xinichai/semantic_base/building_graph.ttl',

                # 输出状态文件路径
                'state_file': '/home/xinichai/semantic_base/components_state.json',

                # 是否显示实时视频窗口
                'show_debug': True
            }]
        ),

        # ====================================================
        # 🧱 JSON 状态同步节点：负责周期检测构件状态完整性
        # ====================================================
        Node(
            package='yolov8_detector',
            executable='json_update_node',
            output='screen',
            prefix=[PYTHON_EXEC, ' '],
            parameters=[{
                'state_file': '/home/xinichai/semantic_base/components_state.json',
                'complete_state': ['installed']#, 'done', 'completed'
            }]
        ),

        # ====================================================
        # 🧠 RDF 语义节点：负责将状态写入 building_graph.ttl
        # ====================================================
        Node(
            package='yolov8_detector',
            executable='rdf_node',
            output='screen',
            prefix=[PYTHON_EXEC, ' '],
            parameters=[{
                # 每30秒同步一次语义信息，与视频检测周期保持一致
                'save_interval': 30,
                'base_rdf': '/home/xinichai/semantic_base/building_graph.ttl',
                'updated_rdf': '/home/xinichai/semantic_base/building_graph_updated.ttl'
            }]
        ),

        # ====================================================
        # 🌐 Flask 可视化节点：网页端动态显示 JSON + RDF
        # ====================================================
        Node(
            package='yolov8_detector',
            executable='viz_node',
            output='screen',
            prefix=[PYTHON_EXEC, ' '],
        ),
    ])


