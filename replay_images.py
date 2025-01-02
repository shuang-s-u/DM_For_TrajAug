import torch
import imageio
import cv2
import os
import numpy as np

def visualize_images(pt_file_path, output_video_path, output_gif_path, fps=20, skip_frames=1):
    """
    从 .pt 文件中提取图像数据并保存为视频和 GIF。

    Args:
        pt_file_path (str): .pt 文件路径。
        output_video_path (str): 保存视频的路径（.mp4）。
        output_gif_path (str): 保存 GIF 的路径。
        fps (int): 每秒帧数（更高的 fps 会使 GIF 播放更快）。
        skip_frames (int): 跳帧间隔，每隔多少帧采样一次。
    """
    # Step 1: 加载 .pt 文件
    data = torch.load(pt_file_path)
    
    # 确保 .pt 文件中有 images 键
    if "images" not in data:
        raise KeyError(f"'images' 键在 {pt_file_path} 中不存在！")

    # 提取图像数据并跳帧
    images = data["images"]
    decompressed_images = images[0: 500][::skip_frames]  # 跳帧采样

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Step 2: 保存为 MP4 视频
    try:
        height, width, channels = decompressed_images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for img in decompressed_images:
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        video_writer.release()
        print(f"视频已保存到: {output_video_path}")
    except Exception as e:
        print(f"保存视频时出错: {e}")
        return

    # Step 3: 从视频生成 GIF
    try:
        # 读取保存的视频帧并生成 GIF
        gif_frames = []
        cap = cv2.VideoCapture(output_video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转回 RGB
        cap.release()

        # 保存为 GIF，并设置更高的 FPS 以加快播放速度
        imageio.mimsave(output_gif_path, gif_frames, fps=fps)
        print(f"GIF 已保存到: {output_gif_path}")
    except Exception as e:
        print(f"生成 GIF 时出错: {e}")

if __name__ == "__main__":
    # 输入 .pt 文件路径和输出路径
    pt_file = "online_logs_ce/quadruped-walk-v0/images_trajectories_step_299000.pt"  # 替换为实际路径
    output_video = "results_ce/quadruped-walk-v0/images_trajectories_step_299000.mp4"
    output_gif = "results_ce/quadruped-walk-v0/images_trajectories_step_299000.gif"
    
    # 可视化图像数据
    visualize_images(pt_file, output_video, output_gif)