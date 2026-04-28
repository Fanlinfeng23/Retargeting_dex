'''
Author: HJX
Date: 2025-04-02 17:52:22
LastEditors: Please set LastEditors
LastEditTime: 2025-04-03 09:45:57
FilePath: /linker_hand_ros2_sdk/src/graphic_display/setup.py
Description: 
symbol_custom_string_obkorol_copyright: 
'''
from setuptools import find_packages, setup

package_name = 'graphic_display'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'linker_hand_ros2_sdk'],
    zip_safe=True,
    maintainer='linker-robot',
    maintainer_email='linker-robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'graphic_display = graphic_display.graphic_display:main'
        ],
    },
)
