from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import torch
import numpy as np
import os
import rospy
from collections import deque
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, CompressedImage
from sdk.msg import RosControl, AlohaCmd, PuppetState
from cv_bridge import CvBridge
import cv2
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Robot Control with Policy Selection') # 解析参数
    parser.add_argument('--policy_type', type=str, default='act',
                      choices=['act', 'diffusion', 'pi0'],
                      help='Type of policy to use (act, diffusion, or pi0)') # 策略类型
    parser.add_argument('--inference_time', type=int, default=60,
                      help='Inference time in seconds') # 推理时间      
    parser.add_argument('--rate', type=float, default=60.0,
                      help='Publishing rate in Hz') # 发布速率
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda, cpu, or mps)') # 设备类型
    parser.add_argument('--ckpt_path', type=str, 
                      default='/root/output_act/checkpoints/020000/pretrained_model',
                      help='Path to the checkpoint file') # 检查点路径
    # 加入temporal_ensemble_coeff
    # parser.add_argument('--temporal_ensemble_coeff', type=float, default=0.9,
    #                   help='Temporal ensemble coefficient')
    return parser.parse_args() # 返回解析的参数

class PolicyFactory:
    @staticmethod
    def create_policy(policy_type, ckpt_path, device):
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

        policy_map = {
            'act': ACTPolicy,
            'diffusion': DiffusionPolicy,
            'pi0': PI0Policy
        }

        if policy_type not in policy_map:
            raise ValueError(f"Unsupported policy type: {policy_type}")

        print("123")
        policy_class = policy_map[policy_type]
        # print(policy_class)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=ckpt_path,
            local_files_only=True,
            map_location=device
        )
        print("456")
        policy.to(device)
        print("1")
        return policy

class RosOperator:
    def __init__(self, rate):
        self._init_queues() # 初始化队列
        self.bridge = CvBridge() # 初始化图像桥接
        self._init_ros(rate) # 初始化ROS

    def _init_queues(self):
        max_size = 10 # 队列最大长度
        self.img_deques = {
            'left': deque(maxlen=max_size), # 左腕图像队列
            'right': deque(maxlen=max_size), # 右腕图像队列
            'front': deque(maxlen=max_size) # 高处图像队列
        }
        self.puppet_state_deque = deque(maxlen=max_size) # 机械臂关节位置队列
        self.gripper_left_deque = deque(maxlen=max_size) # 左夹持器队列
        self.gripper_right_deque = deque(maxlen=max_size) # 右夹持器队列

    def _init_ros(self, rate):
        rospy.init_node('inference_node', anonymous=True)
        self.rate = rospy.Rate(rate)

        # Subscribe to camera topics
        for name, topic in [
            ('left', '/camera_l/color/image_raw/compressed'),
            ('right', '/camera_r/color/image_raw/compressed'),
            ('front', '/camera_f/color/image_raw/compressed')
        ]:
            rospy.Subscriber(
                topic, CompressedImage,
                lambda msg, name=name: self.img_deques[name].append(msg),
                queue_size=1, tcp_nodelay=True
            )

        # Subscribe to state topics
        rospy.Subscriber(
            '/puppet',
            PuppetState,
            lambda msg: self.puppet_state_deque.append(msg),
            queue_size=1
        )
        rospy.Subscriber(
            '/gripper1_position_mm_upsample',
            JointState,
            lambda msg: self.gripper_left_deque.append(msg),
            queue_size=1
        )
        rospy.Subscriber(
            '/gripper2_position_mm_upsample',
            JointState,
            lambda msg: self.gripper_right_deque.append(msg),
            queue_size=1
        )

        # Publishers
        self.aloha_cmd_pub = rospy.Publisher(
            '/aloha_cmd',
            AlohaCmd, 
            queue_size=1
        )
        self.vp_control_pub = rospy.Publisher(
            '/gripper_action_pub',
            JointState,
            queue_size=1
        )

    def get_frame(self):
        """The returned observations do not have a batch dimension."""
        if not all(len(deque) > 0 for deque in [
            self.puppet_state_deque,
            self.gripper_left_deque,
            self.gripper_right_deque,
            *self.img_deques.values()
        ]):
            return None

        obs_dict = {}

        try:
            # Process state information
            puppet_state = self.puppet_state_deque[-1]
            gripper_left = self.gripper_left_deque[-1]
            gripper_right = self.gripper_right_deque[-1]

            """
            # 构建合并的状态和动作向量
                obs_state = torch.cat([
                    state_joint[:7],      # 左臂关节
                    state_effector[:1],   # 左夹持器
                    state_joint[7:],      # 右臂关节
                    state_effector[1:]    # 右夹持器
                ])
            """

            state = []
            # Robot state data
            left_arm_data = np.array(puppet_state.arm_left.position) # 获取机械臂关节位置数据
            right_arm_data = np.array(puppet_state.arm_right.position)
            print(left_arm_data, right_arm_data)

            # left_arm_data = puppet_state_data[:7] # 左臂关节
            # right_arm_data = puppet_state_data[7:] # 右臂关节
            # Gripper data
            left_gripper_data = np.array(gripper_left.position)[:1] # 左夹持器
            right_gripper_data = np.array(gripper_right.position)[:1] # 右夹持器

            state.append(torch.from_numpy(left_arm_data)) # 左臂关节
            state.append(torch.from_numpy(left_gripper_data)) # 左夹持器
            state.append(torch.from_numpy(right_arm_data)) # 右臂关节
            state.append(torch.from_numpy(right_gripper_data)) # 右夹持器

            # Combine all state data
            state = torch.cat(state).float() # 合并状态数据
            obs_dict["observation.state"] = state # 添加到观测字典中

            # Process image data
            for name, deque in self.img_deques.items():
                msg = deque[-1] # 获取最新的图像消息 
                # Decode compressed image
                # 将图像数据转换为numpy数组
                np_arr = np.frombuffer(msg.data, np.uint8) # 将图像数据转换为numpy数组
                print(np_arr)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # 解码图像
                # Convert to torch tensor and ensure float32 type
                img_tensor = torch.from_numpy(img).float() # 将图像转换为torch张量
                obs_dict[f"observation.images.{name}"] = img_tensor # 添加到观测字典中
            # print(obs_dict)
            return obs_dict

        except Exception as e:
            print(f"Error in get_frame: {str(e)}")
            return None

    def publish_actions(self, left_cmd, right_cmd, base_cmd=None):
        aloha_cmd = AlohaCmd()
        aloha_cmd.arx_pos_left = left_cmd[:-1]
        aloha_cmd.arx_pos_right = right_cmd[:-1]
        aloha_cmd.cmd_left = 2
        aloha_cmd.cmd_right = 2


        gripper_msg = JointState()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.position = [
            50.0 if left_cmd[-1] > 35.0 else 0.0,
            50.0 if right_cmd[-1] > 35.0 else 0.0
            # left_cmd[-1],
            # right_cmd[-1]
        ]
        print(gripper_msg.position)
        # Scale gripper values back from [0, 1] to [0, 50]
        # gripper_msg.position = [
        #     left_cmd[-1],   # 反归一化左夹爪
        #     right_cmd[-1]   # 反归一化右夹爪
        # ]

        self.aloha_cmd_pub.publish(aloha_cmd)
        self.vp_control_pub.publish(gripper_msg)
        self.rate.sleep()

def main():
    args = parse_args()

    try:
        # Initialize policy
        policy = PolicyFactory.create_policy(args.policy_type, args.ckpt_path, args.device)

        # Initialize robot
        openloong_robot = RosOperator(args.rate)

        # Main control loop
        total_steps = int(args.inference_time * args.rate*2)
        step_count = 0

        while not rospy.is_shutdown() and step_count < total_steps:
            observation = openloong_robot.get_frame()
            
            # print(f"observation: {observation} \n, type: {type(observation)}, \
            #     keys: {observation.keys()}, \*2*
            #     left_wrist: {observation['observation.images.left_wrist'].shape},\
            #     right_wrist: {observation['observation.images.right_wrist'].shape}, \
            #     high: {observation['observation.images.high'].shape}, \
            #     state: {observation['observation.state'].shape}")

            if observation:
                # Process observation
                for name in observation:
                    if "image" in name:
                        observation[name] = observation[name].type(torch.float32) / 255
                        print(observation[name].shape)
                        observation[name] = observation[name].permute(2, 0, 1).contiguous()
                    observation[name] = observation[name].unsqueeze(0)
                    observation[name] = observation[name].to(args.device)

                # Get action from policy
                # observation shape
                # 计算时间
                time1 = time.time()
                print("1")
                action = policy.select_action(observation)
                print("2")
                action = action.squeeze(0)
                print(f"action: {action}")
                action = action.to("cpu")
                # tensor转numpy
                action = action.detach().numpy()
                action = action.tolist()
                left_cmd, right_cmd = action[:8], action[8:]
                openloong_robot.publish_actions(left_cmd, right_cmd)
                time2 = time.time()
                print(f"<<<<<<<<inference_time<<<<<<<<<{time2 - time1}")
            else:
                print("No Message Observation")
                openloong_robot.rate.sleep()

            step_count += 1

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # 确保程序退出前清理
        rospy.signal_shutdown("Program ended")

if __name__ == "__main__":
    main()

'''
python easy_inference_openloong.py \
    --policy_type act \
    --rate 50.0 \
    --ckpt_path /path/to/checkpoint \
    --device cuda
'''