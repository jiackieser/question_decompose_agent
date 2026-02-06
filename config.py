"""
配置文件 - 包含 Qwen API 配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """应用配置类"""
    
    # Qwen API 配置
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
    QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://vllm-qwen3.vertu.cn/v1")
    QWEN_MODEL = os.getenv("QWEN_MODEL_NAME", "vemory_1_2w_pt")

    # 本地 Qwen API 配置
    QWEN_API_KEY_local = os.getenv("QWEN_API_KEY_LOCAL", "EMPTY")
    QWEN_BASE_URL_local = os.getenv("QWEN_BASE_URL_LOCAL", "https://vllm-qwen3.vertu.cn/v1")
    QWEN_MODEL_local = os.getenv("QWEN_MODEL_LOCAL", "vemory_1_2w_pt")
    
    @classmethod
    def get_qwen_model(cls, temperature: float = 0.7):
        """
        获取 Qwen 模型实例
        
        Args:
            temperature: 温度参数，控制生成文本的随机性
            
        Returns:
            ChatOpenAI: LangChain 兼容的 Qwen 模型实例
        """
        return ChatOpenAI(
            model=cls.QWEN_MODEL,
            api_key=cls.QWEN_API_KEY,
            base_url=cls.QWEN_BASE_URL,
            temperature=temperature,
        )
    
    @classmethod
    def get_qwen_model_local(cls, temperature: float = 0.7):
        """
        获取本地 Qwen 模型实例
        
        Args:
            temperature: 温度参数，控制生成文本的随机性
                
        Returns:
            ChatOpenAI: LangChain 兼容的 Qwen 模型实例
        """
        return ChatOpenAI(
            model=cls.QWEN_MODEL_local,
            api_key=cls.QWEN_API_KEY_local,
            base_url=cls.QWEN_BASE_URL_local,
            temperature=temperature,
        )
