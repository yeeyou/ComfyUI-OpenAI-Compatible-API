"""
OpenAI Compatible API Node for ComfyUI
支持文本和图片输入，调用兼容 OpenAI 的 API
"""

import base64
import io
import json
import requests
from PIL import Image
import torch
import numpy as np


class OpenAICompatibleLLM:
    """
    OpenAI 兼容 API 调用节点
    支持纯文本对话和带图片的视觉对话
    """
    
    def __init__(self):
        self.last_seed = 42
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "hello"
                }),
                "endpoint": ("STRING", {
                    "default": "http://localhost:3010/v1/chat/completions"
                }),
                "model": ("STRING", {
                    "default": ""
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 32000,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 9223372036854775807,
                    "step": 1
                }),
                "seed_control": (["random", "fixed", "increment", "decrement"], {
                    "default": "random"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": ""
                }),
                "image_detail": (["auto", "low", "high"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "OpenAI Compatible"
    
    def tensor_to_base64(self, tensor_image):
        """
        将 ComfyUI 的图片 tensor 转换为 base64 编码的字符串
        ComfyUI 图片格式: [batch, height, width, channels] 范围 0-1
        """
        # 取第一张图片
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]
        
        # 转换为 numpy array 并调整到 0-255 范围
        img_np = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
        
        # 转换为 PIL Image
        img_pil = Image.fromarray(img_np)
        
        # 转换为 base64
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    def generate(self, prompt, endpoint, model, max_tokens, temperature, seed, seed_control,
                 image=None, api_key=None, image_detail="auto"):
        """
        调用 OpenAI 兼容的 API 生成响应
        """
        
        # Handle seed control
        if seed_control == "random":
            import random
            actual_seed = random.randint(0, 9223372036854775807)
        elif seed_control == "fixed":
            actual_seed = seed
        elif seed_control == "increment":
            actual_seed = min(9223372036854775807, self.last_seed + 1)
        elif seed_control == "decrement":
            actual_seed = max(0, self.last_seed - 1)
        else:
            actual_seed = seed
        
        self.last_seed = actual_seed
        
        # 构建消息内容
        if image is not None:
            # 如果有图片输入，使用视觉模型的格式
            image_base64 = self.tensor_to_base64(image)
            
            message_content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64,
                        "detail": image_detail
                    }
                }
            ]
        else:
            # 纯文本对话
            message_content = prompt
        
        # 构建请求体
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": actual_seed
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            # 发送请求
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 提取生成的文本
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
                return (generated_text,)
            else:
                return ("Error: No response from API",)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nDetails: {json.dumps(error_detail, indent=2)}"
                except:
                    error_msg += f"\nResponse: {e.response.text}"
            print(error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return (error_msg,)
