from setuptools import setup
import os
from glob import glob

package_name = 'adaptive_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'models/turtlebot3_waffle_pi_rgbd'),
         glob('models/turtlebot3_waffle_pi_rgbd/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ashish Khanapure',
    maintainer_email='aniruddhanavale232@gmail.com',
    description='DRL adaptive Vision-LiDAR fusion for SLAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_quality_extractor = adaptive_fusion.sensor_quality_extractor:main',
            'fusion_node = adaptive_fusion.fusion_node:main',
            'exploration_controller = adaptive_fusion.exploration_controller:main',
            'odom_noise = adaptive_fusion.odom_noise:main',
            'vehicle_animator = adaptive_fusion.vehicle_animator:main',
        ],
    },
)
