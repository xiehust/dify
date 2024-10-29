import json
from typing import Any, Union

from moviepy.editor import VideoFileClip
import random
import os
from tqdm import tqdm
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from requests.exceptions import RequestException
import requests
from urllib.parse import urlparse
import tempfile
import uuid
from pathlib import Path
import base64
import boto3
from botocore.exceptions import ClientError
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool
import shutil
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            base64_data = base64.b64encode(image_file.read())
            return base64_data.decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None    

def clean_url(url):
    parsed = urlparse(url)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return clean
def image_to_data(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            data = image_file.read()
            return data
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None  
def download_video_with_progress(url,output_dir, filename=None):
    """
    从指定URL下载视频文件到本地临时目录，并显示进度条
    
    参数:
        url (str): 视频文件的URL
        filename (str, optional): 保存的文件名，如果不指定则从URL中获取
        
    返回:
        str: 下载文件的完整路径
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        # 发送 GET 请求获取视频内容
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))

        # 获取原始文件扩展名
        original_ext = Path(clean_url(url)).suffix or '.mp4'  # 如果URL没有扩展名，默认使用.mp4
        if not filename:
            # 生成唯一的临时文件名
            filename = f"{uuid.uuid4()}{original_ext}"

        file_path = os.path.join(output_dir, filename)
        
        # 使用tqdm显示下载进度
        with open(file_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return file_path
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"下载视频失败: {str(e)}")
    
def extract_random_clips(
    video_path, 
    num_clips, 
    clip_duration_minutes, 
    output_dir,
    output_name_prefix="clip",
    min_gap_seconds=0,
    random_seed=None,
    show_progress=True
):
    """
    从视频中随机抽取n段片段并分别保存
    
    参数:
    video_path: 输入视频文件路径
    num_clips: 要抽取的片段数量
    clip_duration_minutes: 每个片段的时长(分钟)
    output_dir: 输出目录路径
    output_name_prefix: 输出文件名前缀
    min_gap_seconds: 片段之间的最小间隔(秒)
    random_seed: 随机数种子，用于复现结果
    show_progress: 是否显示进度条
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到输入视频文件: {video_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载视频文件
        video = VideoFileClip(video_path)
        
        total_duration = video.duration
        clip_duration = clip_duration_minutes * 60
        
        # 检查视频时长是否足够
        if total_duration < clip_duration * num_clips + min_gap_seconds * (num_clips - 1):
            raise ValueError("视频时长不足以提取指定数量和长度的片段")
        
        # 生成不重叠的随机时间点
        start_times = []
        attempts = 0
        max_attempts = 1000  # 防止无限循环
        
        while len(start_times) < num_clips and attempts < max_attempts:
            start_time = random.uniform(0, total_duration - clip_duration)
            
            # 检查是否与已有的片段重叠（考虑最小间隔）
            overlap = False
            for existing_start in start_times:
                if abs(existing_start - start_time) < clip_duration + min_gap_seconds:
                    overlap = True
                    break
            
            if not overlap:
                start_times.append(start_time)
            
            attempts += 1
        
        if len(start_times) < num_clips:
            raise ValueError("无法找到足够的不重叠片段")
        
        # 按时间顺序排序
        start_times.sort()
               
        # 提取并保存片段
        clip_info = []
        if show_progress:
            pbar = tqdm(total=num_clips, desc="处理片段")
        
        for i, start_time in enumerate(start_times, 1):
            # 生成输出文件路径
            output_path = os.path.join(
                output_dir, 
                f"{output_name_prefix}_{i}.mp4"
            )
            
            # 提取片段
            clip = video.subclip(start_time, start_time + clip_duration)
            
            # clip.save_frame(f"{output_path}/frame.png")

            # 保存片段
            clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            
            clip_info.append({
                "clip_number": i,
                "start_time": start_time,
                "duration": clip_duration,
                "output_file": output_path
            })
            
            # 清理资源
            clip.close()
            
            # if show_progress:
            #     pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # 清理资源
        video.close()
        
        return True, {
            "message": "视频片段提取成功",
            "clips": clip_info
        }
        
    except Exception as e:
        return False, {"message": f"处理失败: {str(e)}"}
    


def convert(input_video,output_directory,num_clips=2,clip_duration_minutes=0.1,random_seed=42):

    success, result = extract_random_clips(
        video_path=input_video,
        num_clips=num_clips,
        clip_duration_minutes=clip_duration_minutes,  
        output_dir=output_directory,
        output_name_prefix="random_clip",  # 输出文件名前缀
        min_gap_seconds=5,        # 片段之间至少间隔5秒
        random_seed=random_seed,           # 设置随机种子以复现结果
        show_progress=True        # 显示进度条
    )
    
    if success:
        print(result["message"])
        print("\n提取的片段信息:")
        for clip in result["clips"]:
            print(f"\n片段 {clip['clip_number']}:")
            print(f"  开始时间: {clip['start_time']:.2f} 秒")
            print(f"  时长: {clip['duration']} 秒")
            print(f"  输出文件: {clip['output_file']}")
    else:
        print(result["message"])

def extract_keyframes(
    video_path,
    output_dir,
    method="uniform",
    num_frames=5,
    threshold=0.7,
    min_frame_diff=0.1,
    show_progress=True,
    seed=None 
):
    """
    从视频中提取关键帧
    
    参数:
    video_path: 输入视频路径
    output_dir: 输出目录路径
    method: 提取方法 ('uniform' 或 'difference')
        - uniform: 均匀提取指定数量的帧
        - difference: 基于帧差异提取关键帧
    num_frames: 需要提取的帧数量（仅用于uniform方法）
    threshold: 差异阈值（仅用于difference方法）
    min_frame_diff: 最小帧间隔（秒）
    show_progress: 是否显示进度条
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_frame_interval = int(fps * min_frame_diff)
        
        keyframes = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        if method == "uniform":
            # 均匀提取帧
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            if show_progress:
                pbar = tqdm(total=len(frame_indices), desc="提取关键帧")
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    output_path = os.path.join(
                        output_dir, 
                        f"{video_name}_frame_{frame_idx}.jpg"
                    )
                    cv2.imwrite(output_path, frame)
                    keyframes.append({
                        "frame_idx": frame_idx,
                        "timestamp": frame_idx/fps,
                        "path": output_path
                    })
                    
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
                
        elif method == "random":
            # 设置随机种子
            if seed is not None:
                random.seed(seed)
                
            # 生成可选帧的索引范围（考虑最小间隔）
            available_frames = list(range(0, total_frames, min_frame_interval))
            
            # 如果要提取的帧数大于可用帧数，调整num_frames
            num_frames = min(num_frames, len(available_frames))
            
            # 随机选择帧
            selected_frames = random.sample(available_frames, num_frames)
            selected_frames.sort()  # 按时间顺序排序
            
            if show_progress:
                pbar = tqdm(total=num_frames, desc="提取关键帧")
            
            for frame_idx in selected_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    output_path = os.path.join(
                        output_dir,
                        f"{video_name}_frame_{frame_idx}.jpg"
                    )
                    cv2.imwrite(output_path, frame)
                    keyframes.append({
                        "frame_idx": frame_idx,
                        "timestamp": frame_idx/fps,
                        "path": output_path
                    })
                    
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()     
        elif method == "difference":
            # 基于差异提取帧
            last_keyframe = None
            last_keyframe_idx = -min_frame_interval
            frame_idx = 0
            
            if show_progress:
                pbar = tqdm(total=total_frames, desc="分析帧")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx - last_keyframe_idx >= min_frame_interval:
                    if last_keyframe is None:
                        # 保存第一帧
                        is_keyframe = True
                    else:
                        # 计算与上一个关键帧的差异
                        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        last_gray = cv2.cvtColor(last_keyframe, cv2.COLOR_BGR2GRAY)
                        
                        # 使用SSIM计算相似度
                        score = ssim(current_gray, last_gray)
                        
                        # 如果相似度低于阈值，则认为是关键帧
                        is_keyframe = score < threshold
                        if show_progress:
                            pbar.set_postfix({'SSIM': f'{score:.3f}'})
                    
                    if is_keyframe:
                        output_path = os.path.join(
                            output_dir,
                            f"{video_name}_frame_{frame_idx}.jpg"
                        )
                        cv2.imwrite(output_path, frame)
                        keyframes.append({
                            "frame_idx": frame_idx,
                            "timestamp": frame_idx/fps,
                            "path": output_path,
                            "ssim": score if last_keyframe is not None else 1.0
                        })
                        last_keyframe = frame.copy()
                        last_keyframe_idx = frame_idx
                
                frame_idx += 1
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
                
        else:
            raise ValueError(f"不支持的提取方法: {method}")
        
        # 释放资源
        cap.release()
        
        return True, {
            "message": "关键帧提取成功",
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "keyframes": keyframes
        }
        
    except Exception as e:
        return False, {"message": f"处理失败: {str(e)}"}

# 处理多个视频片段
def process_video_clips(clips_dir, output_base_dir, **kwargs):
    """
    处理目录下的所有视频片段
    
    参数:
    clips_dir: 包含视频片段的目录
    output_base_dir: 关键帧输出的基础目录
    **kwargs: 传递给extract_keyframes的其他参数
    """
    results = []
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(clips_dir)]
    
    for video_file in video_files:
        print(f"processing file:{video_file}")
        video_path = os.path.join(clips_dir, video_file)
        # 为每个视频创建独立的输出目录
        output_dir = os.path.join(
            output_base_dir,
            os.path.splitext(video_file)[0]
        )
        
        success, result = extract_keyframes(
            video_path=video_path,
            output_dir=output_dir,
            **kwargs
        )
        
        results.append({
            "video_file": video_file,
            "success": success,
            "result": result
        })
    
    return results  

def clean_dir(directory):
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        print(f"Folder {directory} does not exist")
    except PermissionError:
        print(f"Permission denied to delete {directory}")

class ExtractVideoFramesTool(BuiltinTool):
    bedrock_client : Any= None
    def _generate_response(self, model_id, message: str) -> str:
        try:
            # Send the message to the model, using a basic inference configuration.
            response = self.bedrock_client.converse(
                modelId=model_id,
                messages=message,
                inferenceConfig={"maxTokens": 2000, "temperature": 0.5, "topP": 0.9},
            )

            # Extract and print the response text.
            response_text = response["output"]["message"]["content"][0]["text"]
            return response_text

        except (ClientError, Exception) as e:
            return(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")

    def _invoke(self, 
                    user_id: str, 
                tool_parameters: dict[str, Any], 
            ) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        
        file_url = tool_parameters.get('file_url')
        num_clips = tool_parameters.get('num_clips',3)
        clip_duration_minutes = tool_parameters.get('clip_duration_minutes',1)
        method = tool_parameters.get('method','random')
        random_seed = tool_parameters.get('random_seed',11)
        num_frames = tool_parameters.get('num_frames',20)
        threshold = tool_parameters.get('threshold',0.96)
        min_frame_diff = tool_parameters.get('min_frame_diff',0.5)
        model_id = tool_parameters.get('model_id',"anthropic.claude-3-5-sonnet-20241022-v2:0")
        instruction = tool_parameters.get('instruction')
                
        # 获取保存路径
        temp_dir = tempfile.gettempdir()
        download_dir = os.path.join(temp_dir,"downloaded_video")
        clips_directory = os.path.join(temp_dir,"output_clips")
        keyframes_base_dir = os.path.join(temp_dir,"all_keyframes")

        try:
            if not self.bedrock_client:
                aws_region = tool_parameters.get('aws_region')
                if aws_region:
                    self.bedrock_client = boto3.client("bedrock-runtime", region_name=aws_region)
                else:
                    self.bedrock_client = boto3.client("bedrock-runtime")


            input_video = download_video_with_progress(file_url,download_dir)
            logger.info(f"Downloaded video to {input_video}")
            
            convert(input_video = input_video,
                    output_directory = clips_directory,
                    num_clips=num_clips,
                    clip_duration_minutes=clip_duration_minutes,
                    random_seed= random_seed)
            
            results = process_video_clips(
                clips_dir=clips_directory,
                output_base_dir=keyframes_base_dir,
                method=method,
                num_frames=num_frames,
                show_progress=False,
                threshold=threshold,
                min_frame_diff=min_frame_diff,
                seed=random_seed
            )
            #读取图片，并返回raw data
            print("\n处理所有视频片段的结果:")
            keyframes  = []
            for result in results:
                frames = []
                if result["success"]:
                    print(f"\n视频 {result['video_file']}:")
                    print(f"提取的关键帧数量: {len(result['result']['keyframes'])}")
                    for keyframe in result['result']['keyframes']:
                        data = image_to_data(keyframe['path'])
                        if data:
                            frames.append(data)
                    keyframes.append(frames)
                else:
                    print(f"\n视频 {result['video_file']} 处理失败:")
                    print(result['result']['message'])


            model_responses = []
            for frames in keyframes:
                images = [{"image":{"format":'png',"source":{"bytes":d}}}  for d in frames]
                conversation = [ { "role": "user",  "content": [{"text": instruction}]+images}]
                resp = self._generate_response(model_id,conversation)
                model_responses.append(resp)

            #clean temp files
            clean_dir(download_dir)
            clean_dir(clips_directory)
            clean_dir(keyframes_base_dir)

            return self.create_text_message("<sep>".join(model_responses))  

        except Exception as e:
            #clean temp files
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                print(f"Folder {temp_dir} does not exist")
            except PermissionError:
                print(f"Permission denied to delete {temp_dir}")
            return self.create_text_message(f'Exception {str(e)}')  

