"""
智能体模块 - 问题分析与拆解智能体（使用 LangChain 官方 Agent API）
"""
import json
from typing import Dict, Any
from config import Config
from tools import ComplexityCheckTool, ProblemDecomposeTool
from react_agent import ReActAgent


class ReActQuestionDecomposeAgent:
    """
    ReAct 框架的问题分析与拆解智能体
    
    使用 LangChain 官方 Agent API，工作流程：
    1. Thought（思考）: 分析当前情况，决定下一步行动
    2. Action（行动）: 选择并执行工具
    3. Observation（观察）: 观察工具返回的结果
    4. 循环: 直到获得足够信息，给出最终答案
    5. Final Answer（最终答案）: 给出最终答案
    """
    
    def __init__(self, temperature: float = 0.3, verbose: bool = False):
        self.agent = ReActAgent(temperature=temperature, verbose=verbose)
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户问题
        
        Args:
            query: 用户原始问题
            
        Returns:
            Dict: 包含分析结果的字典
        """
        try:
            result = self.agent.process(query)
            return result
        except Exception as e:
            return {
                'original_query': query,
                'is_complex': None,
                'sub_problems': [],
                'error': str(e)
            }


class SimpleQuestionDecomposeAgent:
    """
    简化版问题分析与拆解智能体（直接调用工具）
    
    适用于需要更直接控制的场景
    """
    
    def __init__(self):
        self.llm = Config.get_qwen_model(temperature=0.3)
        self.complexity_tool = ComplexityCheckTool()
        self.decompose_tool = ProblemDecomposeTool()
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户问题
        
        Args:
            query: 用户原始问题
            
        Returns:
            Dict: 包含分析结果的字典
        """
        # 第一步：判断复杂度
        print(f"正在分析问题复杂度: {query}")
        complexity_result_str = self.complexity_tool._run(query)
        
        try:
            complexity_result = json.loads(complexity_result_str)
        except json.JSONDecodeError:
            complexity_result = {
                'is_complex': False,
                'reason': '解析失败',
                'indicators': []
            }
        
        is_complex = complexity_result.get('is_complex', False)
        
        # 第二步：如果是复杂问题，进行拆解
        sub_problems = []
        if is_complex:
            print(f"检测到复杂问题，正在进行拆解...")
            decompose_result_str = self.decompose_tool._run(query)
            try:
                decompose_result = json.loads(decompose_result_str)
                sub_problems = decompose_result.get('sub_problems', [])
            except json.JSONDecodeError:
                sub_problems = [{
                    'id': 1,
                    'content': query,
                    'type': '原始问题',
                    'dependencies': []
                }]
        else:
            print(f"简单问题，无需拆解")
            sub_problems = [{
                'id': 1,
                'content': query,
                'type': '简单问题',
                'dependencies': []
            }]
        
        # 返回最终结果
        return {
            'original_query': query,
            'is_complex': is_complex,
            'sub_problems': sub_problems,
            'complexity_analysis': {
                'reason': complexity_result.get('reason', ''),
                'indicators': complexity_result.get('indicators', [])
            }
        }


def create_agent(use_react: bool = True):
    """
    创建智能体实例
    
    Args:
        use_react: 是否使用 ReAct 框架，False 则使用简化版
        
    Returns:
        智能体实例
    """
    if use_react:
        return ReActQuestionDecomposeAgent()
    else:
        return SimpleQuestionDecomposeAgent()
