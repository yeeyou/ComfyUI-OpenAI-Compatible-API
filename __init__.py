"""
ComfyUI OpenAI Compatible API Node
支持调用 OpenAI 兼容的 API，包括文本和图片输入
"""

from .openai_llm_node import OpenAICompatibleLLM

NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleLLM": OpenAICompatibleLLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleLLM": "OpenAI Compatible LLM"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
