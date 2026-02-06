"""
主程序 - 问题分析与拆解智能体的入口
"""
import json
import argparse
from agent import create_agent


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="问题分析与拆解智能体")
    parser.add_argument(
        "--query", 
        "-q", 
        type=str, 
        help="用户输入的问题",
        default=None
    )
    parser.add_argument(
        "--react", 
        "-r", 
        action="store_true", 
        help="使用 ReAct 框架"
    )
    parser.add_argument(
        "--interactive", 
        "-i", 
        action="store_true", 
        help="交互模式"
    )
    
    args = parser.parse_args()
    
    # 创建智能体
    use_react = args.react
    agent = create_agent(use_react=use_react)
    
    print("=" * 60)
    print("问题分析与拆解智能体")
    print(f"使用模式: {'ReAct 框架' if use_react else '简化版'}")
    print("=" * 60)
    
    if args.interactive or args.query is None:
        # 交互模式
        print("\n进入交互模式，输入 'quit' 或 'exit' 退出")
        print("-" * 60)
        
        while True:
            query = input("\n请输入问题: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break
            
            if not query:
                print("请输入有效的问题")
                continue
            
            process_query(agent, query)
    else:
        # 单次查询模式
        process_query(agent, args.query)


def process_query(agent, query: str):
    """
    处理问题并输出结果
    
    Args:
        agent: 智能体实例
        query: 用户问题
    """
    print(f"\n处理中...")
    print("-" * 60)
    
    try:
        result = agent.process(query)
        
        print("\n【分析结果】")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60)
        
        # 打印可读性摘要
        print("\n【摘要】")
        print(f"原始问题: {result['original_query']}")
        print(f"是否复杂: {'是' if result['is_complex'] else '否'}")
        
        if result.get('complexity_analysis'):
            analysis = result['complexity_analysis']
            if analysis.get('reason'):
                print(f"判断理由: {analysis['reason']}")
            if analysis.get('indicators'):
                print(f"复杂特征: {', '.join(analysis['indicators'])}")
        
        print(f"\n子问题数量: {len(result['sub_problems'])}")
        for i, sp in enumerate(result['sub_problems'], 1):
            print(f"  {i}. {sp.get('content', '')}")
            if sp.get('type'):
                print(f"     类型: {sp['type']}")
            if sp.get('dependencies'):
                print(f"     依赖: {sp['dependencies']}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 示例用法
    print("启动问题分析与拆解智能体...")
    print("\n使用示例:")
    print("  python main.py -q 'iPhone 15和华为Mate 60哪个好？'")
    print("  python main.py -i                    # 交互模式")
    print("  python main.py -r -q '如何学习Python？'  # 使用 ReAct 框架")
    print()
    
    main()
