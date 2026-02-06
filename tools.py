"""
工具模块 - 包含问题复杂度判断工具和复杂问题拆解工具
"""
import json
from typing import Dict, List, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config import Config


class ComplexityCheckInput(BaseModel):
    """问题复杂度判断工具的输入参数"""
    query: str = Field(description="用户输入的原始问题")


class ProblemDecomposeInput(BaseModel):
    """复杂问题拆解工具的输入参数"""
    query: str = Field(description="需要拆解的复杂问题")


class ComplexityCheckTool(BaseTool):
    """
    问题复杂度判断工具
    
    """
    
    name: str = "complexity_check"
    description: str = """判断用户问题的复杂度。
    输入：用户原始查询
    输出：{"is_complex": bool, "reason": str, "indicators": list}
    
    复杂问题定义：
    - 包含多个子意图
    - 需要推理或分析
    - 需要多步骤解决
    - 涉及因果关系或对比
    - 多款产品比较
    """
    args_schema: type[BaseModel] = ComplexityCheckInput
    
    def _run(self, query: str) -> str:
        """执行复杂度判断"""
        llm = Config.get_qwen_model(temperature=0.3)
        
        prompt = f"""你是一个问题复杂度分析专家。请判断以下问题是否为复杂问题。

复杂问题的定义（满足以下任一条件即为复杂问题）：
1. 包含多个子意图（如："帮我找北京的酒店和机票"）
2. 需要推理或分析（如："为什么房价会上涨？"）
3. 需要多步骤解决（如："如何从零开始学习Python并找到工作？"）
4. 涉及因果关系或对比（如："比较Python和Java的优缺点"）
5. 多款产品比较（如："iPhone 15和华为Mate 60哪个好？"）

请分析以下问题：
{query}

请以JSON格式输出分析结果，包含以下字段：
- is_complex: true/false，表示是否为复杂问题
- reason: 判断理由（详细说明为什么认为该问题是复杂或简单的）
- indicators: 检测到的复杂特征列表（如["多子意图", "产品比较"]等，简单问题则为空列表）

只输出JSON格式，不要包含其他说明文字。"""
        
        response = llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return json.dumps({
                "is_complex": False,
                "reason": "解析失败，默认为简单问题",
                "indicators": [],
                "raw_response": response.content
            }, ensure_ascii=False, indent=2)
    
    async def _arun(self, query: str) -> str:
        """异步执行（暂未实现）"""
        raise NotImplementedError("异步执行暂未实现")


class ProblemDecomposeTool(BaseTool):
    """
    复杂问题拆解工具
    
    将复杂问题拆解为多个可独立解决的子问题
    """
    
    name: str = "problem_decompose"
    description: str = """将复杂问题拆解为多个子问题。
    输入：需要拆解的复杂问题
    输出：{"sub_problems": [{"id": int, "content": str, "type": str, "dependencies": list}]}
    
    适用于已被判定为复杂的问题，将其拆解为可独立处理的子问题。
    """
    args_schema: type[BaseModel] = ProblemDecomposeInput
    
    def _run(self, query: str) -> str:
        """执行问题拆解"""
        llm = Config.get_qwen_model(temperature=0.5)
        
        prompt = f"""你是一个问题拆解专家。请将以下复杂问题拆解为多个可独立解决的子问题。

需要拆解的问题：
{query}

拆解要求：
1. 每个子问题应该是独立、具体的，可以单独回答
2. 子问题之间可以有依赖关系（用dependencies字段标明）
3. 子问题的顺序应该符合逻辑
4. 确保拆解后的子问题能够完整回答原问题

请以JSON格式输出拆解结果，格式如下：
{{
    "sub_problems": [
        {{
            "id": 1,
            "content": "子问题内容",
            "type": "信息查询/比较分析/推理判断/建议推荐",
            "dependencies": []  // 依赖的子问题ID列表，无依赖则为空
        }}
    ]
}}

只输出JSON格式，不要包含其他说明文字。"""
        
        response = llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return json.dumps({
                "sub_problems": [
                    {
                        "id": 1,
                        "content": query,
                        "type": "原始问题",
                        "dependencies": [],
                        "note": "解析失败，返回原始问题"
                    }
                ],
                "raw_response": response.content
            }, ensure_ascii=False, indent=2)
    
    async def _arun(self, query: str) -> str:
        """异步执行（暂未实现）"""
        raise NotImplementedError("异步执行暂未实现")


def get_tools() -> List[BaseTool]:
    """
    获取所有可用工具
    
    Returns:
        List[BaseTool]: 工具列表
    """
    return [
        ComplexityCheckTool(),
        ProblemDecomposeTool()
    ]
