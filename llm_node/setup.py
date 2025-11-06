from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'llm_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/templates', glob(os.path.join(package_name, 'templates', '*'))),
        ('share/' + package_name + '/static',    glob(os.path.join(package_name, 'static', '*'))),
    ],
    install_requires=[
        'setuptools',
        'Flask',
        'Flask-SocketIO',
        'python-socketio<6',
        'python-engineio<5',
        'requests',
        'sounddevice',
        'numpy',
    ],
    zip_safe=True,
    maintainer='temp_id',
    maintainer_email='kwt9882@naver.com',
    description='ROS2 + Flask-SocketIO web UI node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ros2_chat_node = llm_node.ros2_chat_node:main',
            'llm_inference = llm_node.llm_inference:main',
            'tts_node = llm_node.tts_node:main',
            'control_node = llm_node.control_node:main',
            'stt_node = llm_node.stt_node:main',
            'app_node = llm_node.app_node:main',
        ],
    },
)

