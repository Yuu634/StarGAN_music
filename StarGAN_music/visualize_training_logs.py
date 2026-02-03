#!/usr/bin/env python3
"""
StarGAN_music 学習ログ可視化スクリプト

TensorBoard イベントファイルから学習曲線を抽出し、
matplotlib を使用して可視化します。
画像領域 StarGAN と比較できるように設計されています。

使用方法:
    python visualize_training_logs.py --log_dir ./logs --output_dir ./training_visualization
    python visualize_training_logs.py --compare_with /path/to/stargan/logs
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars_from_event_files(log_dir):
    """
    TensorBoard イベントファイルからスカラー値を抽出
    
    Args:
        log_dir: ログディレクトリのパス
    
    Returns:
        scalars_dict: {metric_name: [(step, value), ...]}
    """
    print(f"\n📂 ログディレクトリ: {log_dir}")
    
    if not os.path.exists(log_dir):
        print(f"  ❌ ディレクトリが存在しません: {log_dir}")
        return None
    
    scalars_dict = defaultdict(list)
    event_files = sorted(Path(log_dir).glob('events.out.tfevents.*'))
    
    print(f"📋 イベントファイル数: {len(event_files)}")
    
    if len(event_files) == 0:
        print(f"  ⚠️ イベントファイルが見つかりません")
        return None
    
    for event_file in event_files:
        print(f"  📖 処理中: {event_file.name}")
        
        try:
            # EventAccumulator を使用してイベントファイルを読み込み
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            # スカラー値を抽出
            scalar_tags = ea.Tags()['scalars']
            print(f"    ✓ スカラータグ数: {len(scalar_tags)}")
            
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                for event in events:
                    # (step, value) のタプルを追加
                    scalars_dict[tag].append((event.step, event.value))
        
        except Exception as e:
            print(f"    ⚠️ 警告: {e}")
            continue
    
    # 各メトリクスをステップでソート
    print(f"📊 抽出されたメトリクス:")
    for tag in sorted(scalars_dict.keys()):
        scalars_dict[tag].sort(key=lambda x: x[0])
        print(f"  ✓ {tag}: {len(scalars_dict[tag])} データ点")
    
    return dict(scalars_dict) if scalars_dict else None


def plot_training_curves(scalars_dict, output_dir, show_plot=True):
    """
    学習曲線を可視化してプロット
    
    Args:
        scalars_dict: スカラー辞書
        output_dir: 出力ディレクトリ
        show_plot: プロット表示フラグ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # メトリクスをカテゴリー別に整理
    metrics_by_category = defaultdict(dict)
    
    for tag, values in scalars_dict.items():
        # タグ形式: "D/loss_real" → category: "D", metric: "loss_real"
        if '/' in tag:
            category, metric = tag.split('/', 1)
        else:
            category = 'Other'
            metric = tag
        
        metrics_by_category[category][metric] = values
    
    print(f"\n📊 メトリクスカテゴリー:")
    for category, metrics in metrics_by_category.items():
        print(f"  {category}: {list(metrics.keys())}")
    
    # ===== Figure 1: Discriminator Losses =====
    if 'D' in metrics_by_category:
        print(f"\n📈 Figure 1: Discriminator Loss")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Discriminator Losses (StarGAN_music)', fontsize=16, fontweight='bold')
        
        d_metrics = metrics_by_category['D']
        
        # Subplot 1: Real/Fake Loss
        ax = axes[0, 0]
        if 'loss_real' in d_metrics:
            steps, values = zip(*d_metrics['loss_real'])
            ax.plot(steps, values, label='D/loss_real', linewidth=2, alpha=0.7, color='blue')
        if 'loss_fake' in d_metrics:
            steps, values = zip(*d_metrics['loss_fake'])
            ax.plot(steps, values, label='D/loss_fake', linewidth=2, alpha=0.7, color='red')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Real / Fake Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Classification Loss
        ax = axes[0, 1]
        if 'loss_cls' in d_metrics:
            steps, values = zip(*d_metrics['loss_cls'])
            ax.plot(steps, values, label='D/loss_cls', color='orange', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 3: GP Loss
        ax = axes[1, 0]
        if 'loss_gp' in d_metrics:
            steps, values = zip(*d_metrics['loss_gp'])
            ax.plot(steps, values, label='D/loss_gp', color='green', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title('Gradient Penalty Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Gradient Penalty Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic', color='red')
        
        # Subplot 4: Gradient Metrics
        ax = axes[1, 1]
        has_grad_data = False
        if 'grad_norm' in d_metrics:
            steps, values = zip(*d_metrics['grad_norm'])
            ax.plot(steps, values, label='D/grad_norm', color='purple', linewidth=2)
            has_grad_data = True
        if 'grad_penalty_norm' in d_metrics:
            steps, values = zip(*d_metrics['grad_penalty_norm'])
            ax.plot(steps, values, label='D/grad_penalty_norm', color='brown', linewidth=2, alpha=0.7)
            has_grad_data = True
        
        if has_grad_data:
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Norm', fontsize=11)
            ax.set_title('Gradient Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Gradient Metrics', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, style='italic')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'discriminator_losses.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 保存: {save_path}")
        if show_plot:
            plt.show()
        plt.close()
    
    # ===== Figure 2: Generator Losses =====
    if 'G' in metrics_by_category:
        print(f"\n📈 Figure 2: Generator Loss")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Generator Losses (StarGAN_music)', fontsize=16, fontweight='bold')
        
        g_metrics = metrics_by_category['G']
        
        # Subplot 1: Fake/Rec/Cls Loss
        ax = axes[0]
        if 'loss_fake' in g_metrics:
            steps, values = zip(*g_metrics['loss_fake'])
            ax.plot(steps, values, label='G/loss_fake', linewidth=2, alpha=0.7)
        if 'loss_rec' in g_metrics:
            steps, values = zip(*g_metrics['loss_rec'])
            ax.plot(steps, values, label='G/loss_rec', linewidth=2, alpha=0.7)
        if 'loss_cls' in g_metrics:
            steps, values = zip(*g_metrics['loss_cls'])
            ax.plot(steps, values, label='G/loss_cls', linewidth=2, alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Total Loss
        ax = axes[1]
        if 'loss' in g_metrics:
            steps, values = zip(*g_metrics['loss'])
            ax.plot(steps, values, label='G/loss_total', color='purple', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'generator_losses.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 保存: {save_path}")
        if show_plot:
            plt.show()
        plt.close()
    
    # ===== Figure 3: D vs G Loss Comparison =====
    print(f"\n📈 Figure 3: D vs G Loss Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Discriminator vs Generator Loss (StarGAN_music)', fontsize=16, fontweight='bold')
    
    if 'D' in metrics_by_category and 'loss' in metrics_by_category['D']:
        steps, values = zip(*metrics_by_category['D']['loss'])
        ax.plot(steps, values, label='D/loss_total', linewidth=2, alpha=0.7, color='blue')
    
    if 'G' in metrics_by_category and 'loss' in metrics_by_category['G']:
        steps, values = zip(*metrics_by_category['G']['loss'])
        ax.plot(steps, values, label='G/loss_total', linewidth=2, alpha=0.7, color='red')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'dg_loss_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  💾 保存: {save_path}")
    if show_plot:
        plt.show()
    plt.close()
    
    # ===== Figure 4: All Metrics in One =====
    print(f"\n📈 Figure 4: All Metrics Overview")
    
    # データポイント数を確認
    num_metrics = len(scalars_dict)
    if num_metrics > 0:
        # メトリクスを3列のグリッドに配置
        ncols = 3
        nrows = (num_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
        fig.suptitle('All Metrics Overview (StarGAN_music)', fontsize=16, fontweight='bold')
        
        # 1Dの場合は2Dに変換
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        
        for idx, (tag, values) in enumerate(sorted(scalars_dict.items())):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            steps, loss_values = zip(*values)
            ax.plot(steps, loss_values, linewidth=2, color='steelblue')
            ax.set_title(tag, fontsize=11, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 余分なサブプロットを非表示
        for idx in range(num_metrics, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'all_metrics_overview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 保存: {save_path}")
        if show_plot:
            plt.show()
        plt.close()


def generate_summary_report(scalars_dict, output_dir):
    """
    学習統計レポートを生成
    
    Args:
        scalars_dict: スカラー辞書
        output_dir: 出力ディレクトリ
    """
    print(f"\n📊 統計レポート生成")
    
    report_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("StarGAN_music Training Summary Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Metrics Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'Count':<10} {'Min':<15} {'Max':<15} {'Mean':<15}\n")
        f.write("-" * 80 + "\n")
        
        for tag in sorted(scalars_dict.keys()):
            values = [v for _, v in scalars_dict[tag]]
            
            if len(values) > 0:
                min_val = np.min(values)
                max_val = np.max(values)
                mean_val = np.mean(values)
                count = len(values)
                
                f.write(f"{tag:<30} {count:<10} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Final Values (Last Recorded):\n")
        f.write("-" * 80 + "\n")
        
        for tag in sorted(scalars_dict.keys()):
            if len(scalars_dict[tag]) > 0:
                last_step, last_value = scalars_dict[tag][-1]
                f.write(f"{tag:<30} Step: {last_step:<10} Value: {last_value:.6f}\n")
    
    print(f"  💾 保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='StarGAN_music Training Log Visualization')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Path to TensorBoard log directory')
    parser.add_argument('--output_dir', type=str, default='./training_visualization',
                       help='Path to output directory for plots')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots (save only)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("StarGAN_music Training Log Visualization")
    print("=" * 80)
    
    # ログ抽出
    scalars_dict = extract_scalars_from_event_files(args.log_dir)
    
    if not scalars_dict:
        print("❌ スカラーデータが見つかりませんでした。")
        print("   ログディレクトリを確認してください。")
        return
    
    # 可視化
    print("\n📊 プロット生成中...")
    plot_training_curves(scalars_dict, args.output_dir, show_plot=not args.no_show)
    
    # レポート生成
    generate_summary_report(scalars_dict, args.output_dir)
    
    print("\n" + "=" * 80)
    print(f"✅ 完了！ 出力ディレクトリ: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

