#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练日志，找出最佳权重文件
目标：同时兼顾低碰撞率和高安全率（Avg overlap 接近 0）
"""

import re
from pathlib import Path
from collections import defaultdict

def parse_log_file(log_path):
    """
    解析训练日志文件

    ⚠️  关键时序说明：
    - Epoch N 的 CVaR验证使用的是 Epoch N-1 训练后的权重
    - Epoch N 的 Eval评估使用的是 Epoch N 训练后的权重
    - 保存的 epoch_N_*.pth 是 Epoch N 训练后的权重

    因此我们需要：
    - 将 Epoch N 的 CVaR ASR 关联到 Epoch N-1 的权重文件
    - Epoch N 的 Eval 指标关联到 Epoch N 的权重文件
    """

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按epoch分割
    epoch_pattern = r'🎯 Epoch (\d+)/\d+'
    epochs = re.split(epoch_pattern, content)

    results = []

    for i in range(1, len(epochs), 2):
        epoch_num = int(epochs[i])
        epoch_content = epochs[i+1]

        # 提取评估指标（这是训练后的性能）
        success_match = re.search(r'Success Rate:\s+([\d.]+)\s+\((\d+)/(\d+)\)', epoch_content)
        collision_match = re.search(r'Collision Rate:\s+([\d.]+)\s+\((\d+)/(\d+)\)', epoch_content)
        reward_match = re.search(r'Average Reward:\s+([-\d.]+)', epoch_content)
        is_best = 'NEW BEST MODEL!' in epoch_content

        # 提取POLAR验证结果中的overlap（这是训练前的性能）
        polar_section = re.search(r'\[3/5\] POLAR 验证 worst-case 轨迹\.\.\.(.*?)⏱️', epoch_content, re.DOTALL)

        overlap_values = []
        unsafe_episodes = []

        if polar_section:
            polar_content = polar_section.group(1)
            # 提取所有的 Avg overlap 值
            overlap_matches = re.findall(r'验证 episode (\d+)\.\.\. ✅ Unsafe: (\d+)/(\d+) steps, Avg overlap: ([\d.]+)', polar_content)

            for ep_num, unsafe_steps, total_steps, overlap in overlap_matches:
                overlap_values.append(float(overlap))
                if float(overlap) > 0:
                    unsafe_episodes.append({
                        'episode': int(ep_num),
                        'unsafe_steps': int(unsafe_steps),
                        'total_steps': int(total_steps),
                        'overlap': float(overlap)
                    })

        if success_match and collision_match and reward_match:
            epoch_data = {
                'epoch': epoch_num,
                # === Eval指标（训练后） ===
                'success_rate': float(success_match.group(1)),
                'success_count': int(success_match.group(2)),
                'total_scenarios': int(success_match.group(3)),
                'collision_rate': float(collision_match.group(1)),
                'collision_count': int(collision_match.group(2)),
                'avg_reward': float(reward_match.group(1)),
                'is_best': is_best,
                # === CVaR指标（训练前） ===
                'cvar_avg_overlap': sum(overlap_values) / len(overlap_values) if overlap_values else None,
                'cvar_max_overlap': max(overlap_values) if overlap_values else None,
                'cvar_min_overlap': min(overlap_values) if overlap_values else None,
                'cvar_zero_overlap_count': sum(1 for ov in overlap_values if ov == 0.0),
                'cvar_total_polar_episodes': len(overlap_values),
                'cvar_unsafe_episodes': unsafe_episodes
            }
            results.append(epoch_data)

    # ✅ 修正：将CVaR指标映射到前一个epoch的权重
    # 首先初始化所有epoch的next_epoch_cvar字段
    for r in results:
        r['next_epoch_cvar_avg_overlap'] = None
        r['next_epoch_cvar_zero_overlap_count'] = None
        r['next_epoch_cvar_total_episodes'] = None
        r['next_epoch_cvar_unsafe_episodes'] = None

    # 然后将下一个epoch的CVaR映射到当前epoch
    for i in range(len(results) - 1):
        # results[i+1]的CVaR反映的是results[i]训练后的性能
        results[i]['next_epoch_cvar_avg_overlap'] = results[i+1]['cvar_avg_overlap']
        results[i]['next_epoch_cvar_zero_overlap_count'] = results[i+1]['cvar_zero_overlap_count']
        results[i]['next_epoch_cvar_total_episodes'] = results[i+1]['cvar_total_polar_episodes']
        results[i]['next_epoch_cvar_unsafe_episodes'] = results[i+1]['cvar_unsafe_episodes']

    return results


def rank_epochs(results):
    """
    根据多个标准对epoch排序

    ⚠️  使用修正后的CVaR指标（next_epoch_cvar）来评估权重文件的真实性能
    """

    # 计算综合得分
    for r in results:
        # ✅ 使用"下一个epoch的CVaR"来评估当前权重的安全性
        if r['next_epoch_cvar_avg_overlap'] is not None:
            # 有CVaR验证数据
            safety_score = r['next_epoch_cvar_zero_overlap_count'] / r['next_epoch_cvar_total_episodes']
            overlap_penalty = r['next_epoch_cvar_avg_overlap']
            has_cvar = True
        else:
            # 没有CVaR数据（例如最后一个epoch），使用默认值
            safety_score = 0.0
            overlap_penalty = 1.0
            has_cvar = False

        # 性能得分：成功率（来自Eval）
        performance_score = r['success_rate']

        # 碰撞惩罚（来自Eval）
        collision_penalty = r['collision_rate']

        # 综合得分 = 0.4 * 安全得分 + 0.4 * 性能得分 - 0.1 * 碰撞惩罚 - 0.1 * overlap惩罚
        r['composite_score'] = (
            0.4 * safety_score +
            0.4 * performance_score -
            0.1 * collision_penalty -
            0.1 * overlap_penalty
        )
        r['safety_score'] = safety_score
        r['has_next_cvar'] = has_cvar

    # 按综合得分排序（只考虑有CVaR数据的epoch）
    epochs_with_cvar = [r for r in results if r['has_next_cvar']]
    ranked = sorted(epochs_with_cvar, key=lambda x: x['composite_score'], reverse=True)

    return ranked


def print_summary(ranked_results, top_n=10):
    """打印总结报告"""

    print("=" * 120)
    print("🏆 训练日志分析结果 - 最佳权重文件推荐（修正时序偏差）")
    print("=" * 120)
    print("\n⚠️  关键修正说明：")
    print("   - Epoch N 保存的权重文件 (epoch_N_*.pth) 是在 Phase 4 训练后保存的")
    print("   - Epoch N 的 CVaR验证发生在训练前，反映的是 Epoch N-1 训练后的权重性能")
    print("   - 因此，我们使用 Epoch N+1 的 CVaR 来评估 Epoch N 权重文件的真实安全性")
    print("\n📊 评估标准说明：")
    print("   1. 安全得分：下一个epoch的CVaR验证中 Avg overlap = 0.000 的比例（越高越好）")
    print("   2. 成功率：当前epoch的Eval成功率（越高越好）")
    print("   3. 碰撞率：当前epoch的Eval碰撞率（越低越好）")
    print("   4. 平均Overlap：下一个epoch的CVaR平均overlap值（越低越好）")
    print("\n" + "=" * 120)
    print(f"\n🥇 Top {top_n} 最佳Epoch（按综合得分排序）：\n")

    print(f"{'Rank':<5} {'Epoch':<6} {'综合得分':<10} {'安全得分':<10} {'成功率':<10} {'碰撞率':<10} "
          f"{'NextCVaR Overlap':<18} {'NextCVaR 0-count':<18} {'标记':<10}")
    print("-" * 120)

    for i, r in enumerate(ranked_results[:top_n], 1):
        best_mark = "⭐ BEST" if r['is_best'] else ""
        if r['has_next_cvar']:
            zero_overlap_ratio = f"{r['next_epoch_cvar_zero_overlap_count']}/{r['next_epoch_cvar_total_episodes']}"
            avg_overlap_str = f"{r['next_epoch_cvar_avg_overlap']:.4f}"
        else:
            zero_overlap_ratio = "N/A"
            avg_overlap_str = "N/A"

        print(f"{i:<5} {r['epoch']:<6} {r['composite_score']:<10.4f} {r['safety_score']:<10.4f} "
              f"{r['success_rate']:<10.3f} {r['collision_rate']:<10.3f} "
              f"{avg_overlap_str:<18} {zero_overlap_ratio:<18} {best_mark:<10}")

    print("\n" + "=" * 120)
    print("\n💎 推荐的权重文件（前5个）：\n")

    for i, r in enumerate(ranked_results[:5], 1):
        print(f"{i}. Epoch {r['epoch']:03d} (综合得分: {r['composite_score']:.4f})")
        print(f"   📂 权重文件: TD3_safety_epoch_{r['epoch']:03d}_*.pth")
        print(f"   📊 Eval性能（训练后）: 成功率={r['success_rate']:.3f}, 碰撞率={r['collision_rate']:.3f}, 平均Reward={r['avg_reward']:.2f}")

        if r['has_next_cvar']:
            asr = r['next_epoch_cvar_zero_overlap_count'] / r['next_epoch_cvar_total_episodes']
            print(f"   🛡️  CVaR安全性（下一个epoch验证）: ASR={asr:.3f} ({r['next_epoch_cvar_zero_overlap_count']}/{r['next_epoch_cvar_total_episodes']} 零overlap), "
                  f"平均overlap={r['next_epoch_cvar_avg_overlap']:.4f}")

            if r['next_epoch_cvar_unsafe_episodes']:
                print(f"   ⚠️  不安全episodes ({len(r['next_epoch_cvar_unsafe_episodes'])}):")
                for ep in r['next_epoch_cvar_unsafe_episodes'][:3]:  # 只显示前3个
                    print(f"      - Episode {ep['episode']}: {ep['unsafe_steps']}/{ep['total_steps']} steps, overlap: {ep['overlap']:.3f}")
            else:
                print(f"   ✅ 所有POLAR验证episodes都是安全的！")
        else:
            print(f"   ⚠️  没有下一个epoch的CVaR数据（可能是最后一个epoch）")
        print()

    print("=" * 120)
    print("\n🎯 完美安全Epoch（下一个epoch的CVaR验证中所有episodes的Avg overlap都为0）：\n")

    perfect_epochs = [r for r in ranked_results if r['has_next_cvar'] and r['next_epoch_cvar_avg_overlap'] == 0.0]

    if perfect_epochs:
        print(f"{'Epoch':<6} {'成功率':<10} {'碰撞率':<10} {'Avg Reward':<12} {'ASR':<8} {'标记':<10}")
        print("-" * 70)
        for r in perfect_epochs[:10]:
            best_mark = "⭐ BEST" if r['is_best'] else ""
            asr = r['next_epoch_cvar_zero_overlap_count'] / r['next_epoch_cvar_total_episodes']
            print(f"{r['epoch']:<6} {r['success_rate']:<10.3f} {r['collision_rate']:<10.3f} "
                  f"{r['avg_reward']:<12.2f} {asr:<8.3f} {best_mark:<10}")
    else:
        print("   未找到所有episodes都完美安全的epoch")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    log_file = "src/drl_navigation_ros2/models/TD3_safety/Dec09_20-20-51_cheeson_from_baseline_frozen/train_output.log"

    print("📖 读取训练日志...")
    results = parse_log_file(log_file)

    print(f"✅ 成功解析 {len(results)} 个epochs\n")

    print("📈 计算综合得分并排序...")
    ranked = rank_epochs(results)

    print_summary(ranked, top_n=15)

    # 保存详细结果到文件
    output_file = "src/drl_navigation_ros2/models/TD3_safety/Dec09_20-20-51_cheeson_from_baseline_frozen/analysis_report.txt"
    import sys
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_summary(ranked, top_n=len(ranked))
    sys.stdout = original_stdout

    print(f"\n💾 详细分析报告已保存到: {output_file}")
