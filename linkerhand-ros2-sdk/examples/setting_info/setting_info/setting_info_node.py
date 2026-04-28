#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
'''
编译: colcon build --symlink-install
启动命令:ros2 run linker_hand_ros2_sdk linker_hand_sdk
'''
import rclpy,math,sys                                     # ROS2 Python接口库
from rclpy.node import Node                      # ROS2 节点类
import rclpy.time
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import JointState
import time,threading, json


class SettingInfoNode(Node):
    def __init__(self):
        super().__init__("setting_info_node")
        self.hand_joint = "L10"
        self.hand_type = "left"
        self.hand_info = {}
        self.cb_count = 0
        self.hand_info_sub = self.create_subscription(String, f"/cb_{self.hand_type}_hand_info",self.setting_cb,10)
        self.setting_pub = self.create_publisher(String, '/cb_hand_setting_cmd', 10)
        

    def setting_cb(self, msg):
        self.cb_count += 1
        self.hand_info = json.loads(msg.data)
        if self.cb_count == 5:
            print(self.hand_info)
            sys.exit(1)
        time.sleep(1)

    
    def pub_msg(self, cmd_dic):
        count = 0
        while True:
            msg = String()
            msg.data = json.dumps(cmd_dic)
            print(msg)
            self.setting_pub.publish(msg)
            time.sleep(1)
            if count == 5:
                break
            count += 1
        

    # 设置速度
    def set_speed(self, speed=[91] * 10):
        cmd_dic = {
            "setting_cmd": "set_speed",
            "params":{
                "hand_type": self.hand_type,
                "speed":speed
            }
        }
        self.pub_msg(cmd_dic=cmd_dic)

    # 设置扭矩
    def set_max_torque_limits(self, torque=[100] * 5):
        cmd_dic = {
            "setting_cmd": "set_max_torque_limits",
            "params":{
                "hand_type": self.hand_type,
                "torque":torque
            }
        }
        self.pub_msg(cmd_dic=cmd_dic)

def main(args=None):
    rclpy.init(args=args)
    set = SettingInfoNode()
    pub_ther = threading.Thread(target=set.set_speed)
    pub_ther.daemon = True
    pub_ther.start()
    #set.set_speed()
    
    rclpy.spin(set)