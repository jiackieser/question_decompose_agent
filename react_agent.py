"""
ReAct Agent 实现 - 使用 LangChain 1.x 官方 API
"""
import json
from typing import Dict, Any
from langchain.agents import create_agent
from config import Config
from tools import get_tools


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent - 使用 LangChain 官方 API
    
    工作流程：
    1. Thought（思考）: 分析当前情况，决定下一步行动
    2. Action（行动）: 选择并执行工具
    3. Observation（观察）: 观察工具返回的结果
    4. 循环: 直到获得足够信息，给出最终答案
    """
    
    def __init__(self, temperature: float = 0.3, verbose: bool = True):
        self.llm = Config.get_qwen_model(temperature=temperature)
        self.tools = get_tools()
        self.agent = self._create_agent(verbose=verbose)
    
    def _create_agent(self, verbose: bool = True):
        """创建 ReAct Agent - 使用 LangChain 1.x API"""
        
        # ReAct 系统提示词
        system_prompt = """你是一个专业的问题分析与拆解助手。你的任务是根据用户的问题，判断其复杂度，并在必要时进行拆解。

你有以下工具可以使用：

{tools}

工具名称: {tool_names}

请按照以下 ReAct 框架进行思考和工作：

1. Thought（思考）: 分析当前情况，决定下一步行动
2. Action（行动）: 选择要使用的工具，格式为：Action: 工具名称, 输入: 工具参数
3. Observation（观察）: 观察工具返回的结果
4. 重复以上步骤直到获得足够信息
5. Final Answer（最终答案）: 给出最终答案


重要提示：
- 每个 Action 必须是有效的格式
- 如果问题已经是简单问题（is_complex=false），则不需要拆解，直接输出
- 最终答案必须是 JSON 格式，包含以下字段：
  - original_query: 原始问题
  - is_complex: 是否为复杂问题（true/false）
  - sub_problems: 子问题列表（如果是复杂问题）
  - complexity_analysis: 复杂度分析（reason 和 indicators）

开始工作！"""

        # 准备工具描述
        tools_description = "\n\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_names = ", ".join([tool.name for tool in self.tools])
        
        # 格式化系统提示词
        formatted_system_prompt = system_prompt.format(
            tools=tools_description,
            tool_names=tool_names
        )
        
        # 使用 LangChain 1.x 的 create_agent API
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=formatted_system_prompt
        )
        
        return agent
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户问题 - 使用 LangChain 1.x API
        
        Args:
            query: 用户原始问题
            
        Returns:
            Dict: 处理结果
        """
        try:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            # 从结果中提取输出
            output = ""
            if isinstance(result, dict):
                output = result.get("output", "")
                if not output and "messages" in result:
                    messages = result["messages"]
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, "content"):
                            output = last_message.content
                        elif isinstance(last_message, dict):
                            output = last_message.get("content", "")
            elif hasattr(result, "content"):
                output = result.content
            
            # 尝试解析最终输出为 JSON
            try:
                # 尝试从输出中提取 Final Answer 后的 JSON
                final_answer_marker = "Final Answer:"
                if final_answer_marker in output:
                    final_answer_start = output.find(final_answer_marker) + len(final_answer_marker)
                    final_answer_text = output[final_answer_start:].strip()
                    
                    # 提取 Final Answer 中的 JSON
                    json_start = final_answer_text.find("{")
                    json_end = final_answer_text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = final_answer_text[json_start:json_end]
                        parsed_result = json.loads(json_str)
                        return self._format_final_result(query, parsed_result)
                else:
                    # 如果没有 Final Answer 标记，尝试直接提取 JSON
                    json_start = output.find("{")
                    json_end = output.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = output[json_start:json_end]
                        parsed_result = json.loads(json_str)
                        return self._format_final_result(query, parsed_result)
            except json.JSONDecodeError:
                pass
            
            # 如果解析失败，返回原始输出
            return {
                "original_query": query,
                "is_complex": None,
                "sub_problems": [],
                "raw_output": output
            }
            
        except Exception as e:
            return {
                "original_query": query,
                "is_complex": None,
                "sub_problems": [],
                "error": str(e)
            }
    
    def _format_final_result(self, query: str, result: Dict) -> Dict[str, Any]:
        """
        格式化最终结果
        
        Args:
            query: 原始问题
            result: 解析后的结果
            
        Returns:
            Dict: 标准化格式的结果
        """
        # 提取复杂度信息
        is_complex = result.get("is_complex", False)
        if isinstance(is_complex, str):
            is_complex = is_complex.lower() == "true"
        
        # 提取子问题列表
        sub_problems = []
        if "sub_problems" in result:
            sub_problems = result["sub_problems"]
        elif is_complex:
            # 如果是复杂问题但没有子问题，将整个问题作为一个子问题
            sub_problems = [{
                "id": 1,
                "content": query,
                "type": "原始问题",
                "dependencies": []
            }]
        
        return {
            "original_query": query,
            "is_complex": is_complex,
            "sub_problems": sub_problems,
            "complexity_analysis": {
                "reason": result.get("reason", ""),
                "indicators": result.get("indicators", [])
            }
        }
