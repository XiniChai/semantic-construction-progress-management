from setuptools import find_packages, setup
import os
import sys
from glob import glob


# 强制指定虚拟环境 python
VENV_PYTHON = "/home/xinichai/yolo_infer_env/bin/python"
if os.path.exists(VENV_PYTHON):
    sys.executable = VENV_PYTHON


package_name = 'yolov8_detector'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pipeline.launch.py']),
        #launch文件夹被安装
        #(os.path.join('share', package_name,'launch'),glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xinichai',
    maintainer_email='xinichai@todo.todo',
    description='YOLOv8 inference + semantic pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'yolo_node = yolov8_detector.yolo_node:main',
            'step_filter_node = yolov8_detector.step_filter_node:main',
            'step_filter_node_video = yolov8_detector.step_filter_node_video:main',
            'json_update_node = yolov8_detector.json_update_node:main',
            'rdf_node = yolov8_detector.rdf_node:main',
            'viz_node = yolov8_detector.viz_node:main',
        ],
    },
    # 🔑 这里强制所有节点脚本用虚拟环境 python
    options={
        'build_scripts': {
            'executable': '/home/xinichai/yolo_infer_env/bin/python'
        }
    }
)
