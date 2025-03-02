#!/usr/bin/env python
# TODO 需要修改代码逻辑 让代码可以适配多文件结构
import os
import h5py
import json
import torch
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
from datetime import datetime
from typing import Dict, Any, List
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def create_robot_features():
    """
    创建机器人数据特征定义，包含两类数据格式：
    1. 详细的机器人控制数据：分别存储每个关节和夹持器的状态与动作
    2. 用于可视化的合并数据：将状态和动作数据整合为统一的向量
    
    数据维度说明：
    - 机器人关节: 每臂7个关节，共14维
    - 夹持器: 左右各1个，共2维
    - 合并后的状态和动作向量: 16维 (7个关节 + 1个夹持器 + 7个关节 + 1个夹持器)
    - 相机图像: 640x480的RGB图像
    
    Returns:
        Dict: 包含所有特征定义的字典
    """
    features = {
        # 详细的机器人状态数据 - 用于精确控制和分析
        "state_joint_position": {
            "dtype": "float64",
            "shape": [14],  # 左右臂各7个关节
            "names": ["left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4",
                     "left_arm_5", "left_arm_6", "left_arm_7",
                     "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", 
                     "right_arm_5", "right_arm_6", "right_arm_7"]
        },
        "state_effector_position": {
            "dtype": "float64", 
            "shape": [2],  # 左右夹持器
            "names": ["left_gripper", "right_gripper"]
        },
        
        # 详细的机器人动作数据 - 用于精确控制和分析
        "action_joint_position": {
            "dtype": "float64",
            "shape": [14],
            "names": ["left_arm_exp_1", "left_arm_exp_2", "left_arm_exp_3", "left_arm_exp_4",
                     "left_arm_exp_5", "left_arm_exp_6", "left_arm_exp_7",
                     "right_arm_exp_1", "right_arm_exp_2", "right_arm_exp_3", "right_arm_exp_4",
                     "right_arm_exp_5", "right_arm_exp_6", "right_arm_exp_7"]
        },
        "action_effector_position": {
            "dtype": "float64",
            "shape": [2],
            "names": ["left_gripper_exp", "right_gripper_exp"]
        },

        # 合并后的状态和动作数据 - 用于可视化
        "observation.state": {
            "dtype": "float64",
            "shape": [16],  # 14个关节 + 2个夹持器
            "names": ["left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4",
                     "left_arm_5", "left_arm_6", "left_arm_7", "left_gripper",
                     "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", 
                     "right_arm_5", "right_arm_6", "right_arm_7", "right_gripper"]
        },
        "action": {
            "dtype": "float64",
            "shape": [16],  # 14个关节 + 2个夹持器
            "names": ["left_arm_exp_1", "left_arm_exp_2", "left_arm_exp_3", "left_arm_exp_4",
                     "left_arm_exp_5", "left_arm_exp_6", "left_arm_exp_7", "left_gripper_exp",
                     "right_arm_exp_1", "right_arm_exp_2", "right_arm_exp_3", "right_arm_exp_4",
                     "right_arm_exp_5", "right_arm_exp_6", "right_arm_exp_7", "right_gripper_exp"]   
        },

        # 相机图像数据定义
        "observation.images.front": {
            "dtype": "video",  
            "shape": [3, 480, 640],  
            "names": ["channel", "height", "width"]  # 指定维度名称
        },
        "observation.images.left": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"]  # 指定维度名称
        },
        "observation.images.right": {
            "dtype": "video", 
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"]  # 指定维度名称
        }
    }
    return features

def compute_data_stats(data: np.ndarray) -> Dict[str, float]:
    """
    计算数据的基本统计信息，用于数据集的可视化和分析。
    这些统计信息帮助理解数据的分布特征，对于后续的数据标准化很有帮助。
    
    Args:
        data: 输入数据数组
        
    Returns:
        包含统计信息的字典：均值、标准差、最小值、最大值
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }

def process_image(img_path: str) -> torch.Tensor:
    """
    处理图像以符合 LeRobot 数据集的要求
    
    步骤：
    1. 读取图像并转换为 RGB
    2. 转换为 numpy 数组
    3. 将数值范围归一化到 [0, 1]
    4. 将格式从 [H, W, C] 转换为 [C, H, W]
    5. 转换为 float32 类型
    
    Args:
        img_path: 图像文件路径
    
    Returns:
        符合要求的 PyTorch 张量
    """
    # 读取并转换为 RGB
    img = Image.open(img_path)
    img_rgb = img.convert('RGB')
    
    # 转换为 numpy 数组并归一化
    img_array = np.array(img_rgb).astype(np.float32) / 255.0
    
    # 转换为 PyTorch 张量并调整通道顺序
    img_tensor = torch.from_numpy(img_array)
    img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    return img_tensor

def process_image_batch(frame_indices: List[int], input_dir: str, cameras: List[str]) -> List[Dict]:
    """
    批量处理多帧图像数据，使用多线程加速处理。
    
    修改后的工作流程：
    1. 根据新的目录结构构建图像路径
    2. 并行读取和处理多个图像
    3. 整理结果并返回
    
    Args:
        frame_indices: 要处理的帧索引列表
        input_dir: 输入的camera目录路径，例如 ".../final_output/20250211_115849/camera"
        cameras: 相机列表 ['front', 'left', 'right']
        
    Returns:
        处理后的图像数据列表，每个元素是一帧的所有相机图像
    """
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        
        # 提交所有图像处理任务，注意这里路径构建方式的改变
        for frame_idx in frame_indices:
            for cam in cameras:
                # 新的图像路径构建方式
                img_path = os.path.join(input_dir, str(frame_idx), f'cam_{cam}_color.jpg')
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"图像文件不存在: {img_path}")
                futures.append(executor.submit(process_image, img_path))
        
        # 收集并整理结果
        for i, future in enumerate(futures):
            frame_idx = frame_indices[i // len(cameras)]
            cam = cameras[i % len(cameras)]
            if len(results) <= i // len(cameras):
                results.append({})
            results[i // len(cameras)][f'observation.images.{cam}'] = future.result()
    
    return results

def convert_to_lerobot(input_dir: str, output_dir: str, repo_id: str, fps: int = 10):
    """
    将时间戳格式的输入数据转换为标准化的LeRobot数据集格式
    
    数据目录结构说明：
    输入：
        final_output/
            20250211_115849/
                camera/
                    0/
                        cam_front_color.jpg
                        cam_left_color.jpg
                        cam_right_color.jpg
                    1/
                        ...
                record/
                    aligned_joints.h5
            20250211_120005/
                ...
    
    Args:
        input_dir: 输入数据根目录
        output_dir: 输出目录
        repo_id: 数据集的唯一标识符
        fps: 数据采样帧率
    """
    print(f"Starting conversion from {input_dir}")
    
    # 创建数据集实例
    features = create_robot_features()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        features=features,
        use_videos=True
    )
    
    # 启用异步图像写入
    dataset.start_image_writer(num_processes=10, num_threads=10)
    
    # 查找所有时间戳子目录
    timestamp_dirs = []
    for entry in os.scandir(os.path.join(input_dir)):  # 移除了 "final_output" 路径段
        if entry.is_dir() and entry.name.startswith("202"):
            timestamp_dirs.append(entry.path)
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"未找到有效的时间戳数据目录在: {input_dir}")
    
    print(f"找到 {len(timestamp_dirs)} 个数据会话目录")
    
    

    done_list = []
    # 处理每个时间戳目录
    for session_dir in sorted(timestamp_dirs):
        session_id = os.path.basename(session_dir)
        print(f"\n处理数据会话: {session_id}")
        
        # 验证输入文件
        h5_path = os.path.join(session_dir, "record", "aligned_joints.h5")
        if not os.path.exists(h5_path):
            print(f"警告: 会话 {session_id} 中未找到H5文件，跳过")
            continue
        
        print(f"读取关节数据: {h5_path}")
        stats = {}
        
        with h5py.File(h5_path, 'r') as h5f:
            # 将关节数据预加载到内存
            print("加载关节数据到内存...")
            joint_data = {
                'state/joint/position': h5f['state/joint/position'][:],
                'state/effector/position': h5f['state/effector/position'][:],
                'action/joint/position': h5f['action/joint/position'][:],
                'action/effector/position': h5f['action/effector/position'][:]
            }
            
            # 计算统计信息
            for key, data in joint_data.items():
                stats[f"{session_id}/{key.replace('/', '_')}"] = compute_data_stats(data)
            
            total_frames = len(joint_data['state/joint/position'])
            print(f"找到 {total_frames} 帧待处理")
            
            # 创建新的episode缓冲区
            episode_buffer = dataset.create_episode_buffer()
            
            # 批量处理数据
            batch_size = 64
            for batch_start in tqdm(range(0, total_frames, batch_size), desc="处理数据批次"):
                batch_end = min(batch_start + batch_size, total_frames)
                frame_indices = list(range(batch_start, batch_end))
                
                # 处理图像数据session_dir
                image_results = process_image_batch(
                    frame_indices=frame_indices,
                    input_dir=os.path.join(session_dir, "camera"),  # 更新图像路径
                    cameras=['front', 'left', 'right']
                )
                
                # 处理每一帧数据
                for i, frame_idx in enumerate(frame_indices):
                    # 准备详细的状态和动作数据
                    state_joint = torch.from_numpy(joint_data['state/joint/position'][frame_idx]).float()
                    state_effector = torch.from_numpy(joint_data['state/effector/position'][frame_idx]).float()
                    action_joint = torch.from_numpy(joint_data['action/joint/position'][frame_idx]).float()
                    action_effector = torch.from_numpy(joint_data['action/effector/position'][frame_idx]).float()

                    # 构建合并的状态和动作向量
                    obs_state = torch.cat([
                        state_joint[:7],      # 左臂关节
                        state_effector[:1],   # 左夹持器
                        state_joint[7:],      # 右臂关节
                        state_effector[1:]    # 右夹持器
                    ])
                    action = torch.cat([
                        action_joint[:7],     # 左臂目标关节
                        action_effector[:1],  # 左夹持器目标
                        action_joint[7:],     # 右臂目标关节
                        action_effector[1:]   # 右夹持器目标
                    ])
                    
                    # 组合所有数据
                    frame_data = {
                        "state_joint_position": state_joint,
                        "state_effector_position": state_effector,
                        "action_joint_position": action_joint,
                        "action_effector_position": action_effector,
                        "observation.state": obs_state,
                        "action": action,
                    }
                    
                    # 添加图像数据
                    frame_data.update(image_results[i])
                    
                    # 添加帧到数据集
                    dataset.add_frame(frame_data)
            
            # 保存当前会话的episode
            print(f"\n保存会话 {session_id} 的episode...")
            dataset.save_episode(
                task=f"robot manipulation task - session {session_id}", 
                encode_videos=True
            )
        done_list.append(session_dir)
    
    
    # 保存累积的统计信息
    stats_path = os.path.join(output_dir, "meta", "stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 停止异步图像写入
    dataset.stop_image_writer()
    
    # 整理数据集
    print("整理数据集...")
    dataset.consolidate()
    
    print(f"数据集转换完成。输出保存到: {output_dir}")
    
def main():
    """
    主函数：解析命令行参数并执行数据转换。
    提供了灵活的参数配置，方便用户根据需要调整转换过程。
    """
    parser = argparse.ArgumentParser(description='Convert synchronized data to LeRobot format')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing synchronized data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for LeRobot dataset')
    parser.add_argument('--repo_id', type=str, required=True,
                      help='Repository ID for the dataset')
    parser.add_argument('--fps', type=int, default=10,
                      help='Frames per second of the dataset')
    
    args = parser.parse_args()
    
    # 执行转换
    convert_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps
    )

if __name__ == '__main__':
    main()

# ln -s /Users/kenton/lerobot/data/openloong_t1_multi /Users/kenton/.cache/huggingface/lerobot/kenton/openloong_t1_multi