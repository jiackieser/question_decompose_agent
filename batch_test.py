"""
批量测试脚本 - 读取测试用例，运行智能体预测，计算准确率
使用 ReAct Agent 进行复杂度判定
"""
import csv
import json
import time
import datetime
from typing import List, Dict, Tuple
from react_agent import ReActAgent


def load_test_cases(csv_file: str) -> List[Dict]:
    """
    加载测试用例
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        List[Dict]: 测试用例列表
    """
    test_cases = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # 打印列名用于调试
        print(f"CSV列名: {reader.fieldnames}")
        
        for row in reader:
            # 去除列名中的空格和BOM
            row_cleaned = {k.strip().replace('\ufeff', ''): v for k, v in row.items()}
            
            try:
                test_cases.append({
                    'question': row_cleaned.get('question', ''),
                    'is_complexity': int(row_cleaned.get('is_complexity', 0)),
                    'human_eval': int(row_cleaned.get('hunman_eval', 0))
                })
            except (KeyError, ValueError) as e:
                print(f"解析行出错: {e}, 行数据: {row_cleaned}")
                continue
    return test_cases


def run_batch_test(test_cases: List[Dict], delay: float = 0.5) -> Tuple[List[Dict], float]:
    """
    批量运行测试 - 使用 ReAct Agent 进行复杂度判定
    
    Args:
        test_cases: 测试用例列表
        delay: 每次请求之间的延迟（秒）
        
    Returns:
        Tuple[List[Dict], float]: (结果列表, 准确率)
    """
    # 用 ReAct Agent
    agent = ReActAgent(temperature=0.3, verbose=False)
    results = []
    correct_count = 0
    total_count = len(test_cases)
    
    print(f"开始批量测试，共 {total_count} 个测试用例...")
    print("=" * 80)
    
    for i, test_case in enumerate[Dict](test_cases, 1):
        question = test_case['question']
        expected = test_case['human_eval']
        
        print(f"\n[{i}/{total_count}] 测试问题: {question}")
        print(f"预期结果: {'复杂问题' if expected == 1 else '简单问题'}")
        
        try:
            # 使用 ReAct Agent 处理问题
            agent_result = agent.process(question)
            
            # 从 Agent 返回结果中提取复杂度信息
            is_complex = agent_result.get('is_complex', False)
            reason = ''
            indicators = []
            
            # 尝试从 complexity_analysis 中提取理由和指标
            if 'complexity_analysis' in agent_result:
                analysis = agent_result['complexity_analysis']
                reason = analysis.get('reason', '')
                indicators = analysis.get('indicators', [])
            
            # 转换为 0/1
            predicted = 1 if is_complex else 0
            
            # 判断是否正确
            is_correct = (predicted == expected)
            if is_correct:
                correct_count += 1
            
            # 保存结果
            result_record = {
                'question': question,
                'expected': expected,
                'predicted': predicted,
                'is_correct': is_correct,
                'is_complex': is_complex,
                'reason': reason,
                'indicators': ','.join(indicators) if indicators else ''
            }
            results.append(result_record)
            
            print(f"预测结果: {'复杂问题' if predicted == 1 else '简单问题'} {'✓' if is_correct else '✗'}")
            if reason:
                print(f"判断理由: {reason[:50]}...")
            
            # 延迟，避免请求过快
            if i < total_count:
                time.sleep(delay)
                
        except Exception as e:
            print(f"处理出错: {e}")
            results.append({
                'question': question,
                'expected': expected,
                'predicted': None,
                'is_correct': False,
                'error': str(e)
            })
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"测试完成！")
    print(f"总样本数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.2%}")
    
    return results, accuracy


def save_results(results: List[Dict], output_file: str):
    """
    保存测试结果到CSV
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question', 'expected', 'predicted', 'is_correct', 
                      'is_complex', 'reason', 'indicators']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'question': result['question'],
                'expected': result['expected'],
                'predicted': result['predicted'],
                'is_correct': result['is_correct'],
                'is_complex': 1 if result.get('is_complex', False) else 0,
                'reason': result.get('reason', ''),
                'indicators': result.get('indicators', '')
            })
    print(f"\n结果已保存到: {output_file}")


def save_accuracy(accuracy: float, output_file: str):
    """
    保存准确率到文件
    
    Args:
        accuracy: 准确率（0-1之间的浮点数）
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"准确率: {accuracy:.2%}\n")
        f.write(f"数值: {accuracy:.4f}\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"准确率已保存到: {output_file}")


def print_confusion_matrix(results: List[Dict]):
    """
    打印混淆矩阵
    
    Args:
        results: 结果列表
    """
    tp = sum(1 for r in results if r['expected'] == 1 and r['predicted'] == 1)
    tn = sum(1 for r in results if r['expected'] == 0 and r['predicted'] == 0)
    fp = sum(1 for r in results if r['expected'] == 0 and r['predicted'] == 1)
    fn = sum(1 for r in results if r['expected'] == 1 and r['predicted'] == 0)
    
    print("\n" + "=" * 80)
    print("混淆矩阵:")
    print("-" * 40)
    print(f"                 预测")
    print(f"              简单    复杂")
    print(f"实际  简单    {tn:4d}    {fp:4d}")
    print(f"      复杂    {fn:4d}    {tp:4d}")
    print("-" * 40)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率 (Precision): {precision:.2%}")
    print(f"召回率 (Recall): {recall:.2%}")
    print(f"F1分数: {f1:.2%}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量测试问题复杂度判断智能体')
    parser.add_argument('--input', '-i', type=str, default='100_test_samples.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default='test_results_1.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--accuracy-output', '-a', type=str, default=None,
                        help='准确率输出文件路径（默认为 output + _accuracy.txt）')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                        help='请求间隔延迟（秒）')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='限制测试样本数量（用于快速测试）')
    
    args = parser.parse_args()
    
    # 加载测试用例
    print(f"加载测试用例: {args.input}")
    test_cases = load_test_cases(args.input)
    print(f"共加载 {len(test_cases)} 个测试用例")
    
    # 限制样本数量（如果指定）
    if args.limit and args.limit > 0:
        test_cases = test_cases[:args.limit]
        print(f"限制测试样本数为: {args.limit}")
    
    # 运行批量测试
    results, accuracy = run_batch_test(test_cases, delay=args.delay)
    
    # 打印混淆矩阵
    # print_confusion_matrix(results)
    
    # 保存结果
    save_results(results, args.output)
    
    # 保存准确率
    if args.accuracy_output:
        accuracy_file = args.accuracy_output
    else:
        accuracy_file = args.output.replace('.csv', '_accuracy.txt')
    save_accuracy(accuracy, accuracy_file)
    
    # 打印错误样本
    print("\n" + "=" * 80)
    print("预测错误的样本:")
    print("-" * 80)
    error_count = 0
    for result in results:
        if not result['is_correct']:
            error_count += 1
            print(f"\n{error_count}. 问题: {result['question']}")
            print(f"   预期: {'复杂' if result['expected'] == 1 else '简单'}", end='')
            print(f", 预测: {'复杂' if result['predicted'] == 1 else '简单'}")
            if result.get('reason'):
                print(f"   理由: {result['reason'][:80]}...")
    
    if error_count == 0:
        print("无错误样本！")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
