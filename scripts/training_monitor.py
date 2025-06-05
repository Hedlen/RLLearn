#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控和检查点管理工具
"""

import os
import json
import time
import torch
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

class TrainingMonitor:
    """训练进度监控器"""
    
    def __init__(self, output_dir: str = "./output", log_file: str = "training_log.json"):
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / log_file
        self.start_time = None
        self.metrics_history = []
        self.step_times = []
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_training(self, total_steps: int, model_name: str = ""):
        """开始训练监控"""
        self.start_time = time.time()
        self.total_steps = total_steps
        self.model_name = model_name
        
        # 记录训练开始信息
        start_info = {
            "start_time": datetime.now().isoformat(),
            "total_steps": total_steps,
            "model_name": model_name,
            "status": "started"
        }
        
        self._save_log(start_info)
        print(f"🚀 开始训练监控: {model_name}")
        print(f"   总步数: {total_steps:,}")
        print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_step(self, step: int, metrics: Dict[str, float], step_time: float = None):
        """记录训练步骤"""
        current_time = time.time()
        
        if step_time is None:
            if len(self.step_times) > 0:
                step_time = current_time - self.step_times[-1]
            else:
                step_time = current_time - self.start_time if self.start_time else 0
        
        self.step_times.append(current_time)
        
        # 记录指标
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": current_time - self.start_time if self.start_time else 0,
            "step_time": step_time,
            "metrics": metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # 计算进度和ETA
        progress = step / self.total_steps if hasattr(self, 'total_steps') else 0
        eta = self._estimate_eta(step)
        
        # 显示进度
        if step % 10 == 0 or step == 1:  # 每10步显示一次
            self._print_progress(step, progress, eta, metrics, step_time)
        
        # 定期保存日志
        if step % 100 == 0:
            self._save_current_state()
    
    def _estimate_eta(self, current_step: int) -> str:
        """估算剩余时间"""
        if not hasattr(self, 'total_steps') or current_step == 0:
            return "未知"
        
        # 使用最近的步骤时间计算平均速度
        recent_steps = min(50, len(self.step_times))
        if recent_steps < 2:
            return "计算中..."
        
        recent_times = self.step_times[-recent_steps:]
        avg_step_time = (recent_times[-1] - recent_times[0]) / (recent_steps - 1)
        
        remaining_steps = self.total_steps - current_step
        remaining_seconds = remaining_steps * avg_step_time
        
        return self._format_time(remaining_seconds)
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.0f}分钟"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}小时{minutes:.0f}分钟"
    
    def _print_progress(self, step: int, progress: float, eta: str, metrics: Dict[str, float], step_time: float):
        """打印训练进度"""
        # 进度条
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 格式化指标
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        print(f"\r步骤 {step:,}/{getattr(self, 'total_steps', '?'):,} [{bar}] {progress*100:.1f}% | "
              f"ETA: {eta} | 步时: {step_time:.2f}s | {metrics_str}", end="", flush=True)
        
        # 每100步换行
        if step % 100 == 0:
            print()
    
    def _save_log(self, data: Dict):
        """保存日志"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存日志失败: {e}")
    
    def _save_current_state(self):
        """保存当前状态"""
        state = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "total_steps": getattr(self, 'total_steps', 0),
            "model_name": getattr(self, 'model_name', ""),
            "current_step": len(self.metrics_history),
            "metrics_history": self.metrics_history[-100:],  # 只保存最近100步
            "last_update": datetime.now().isoformat()
        }
        
        self._save_log(state)
    
    def finish_training(self, final_metrics: Dict[str, float] = None):
        """结束训练监控"""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        print(f"\n\n🎉 训练完成!")
        print(f"   总耗时: {self._format_time(total_time)}")
        print(f"   总步数: {len(self.metrics_history):,}")
        
        if final_metrics:
            print("   最终指标:")
            for k, v in final_metrics.items():
                print(f"     {k}: {v:.4f}")
        
        # 保存最终状态
        final_state = {
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "total_time": total_time,
            "total_steps": len(self.metrics_history),
            "final_metrics": final_metrics or {},
            "metrics_history": self.metrics_history
        }
        
        self._save_log(final_state)
        
        # 生成训练报告
        self.generate_report()
    
    def generate_report(self):
        """生成训练报告"""
        if not self.metrics_history:
            print("⚠️  没有训练数据，无法生成报告")
            return
        
        try:
            # 提取指标数据
            steps = [entry['step'] for entry in self.metrics_history]
            
            # 获取所有指标名称
            all_metrics = set()
            for entry in self.metrics_history:
                all_metrics.update(entry['metrics'].keys())
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'训练报告 - {getattr(self, "model_name", "Unknown Model")}', fontsize=16)
            
            # 损失曲线
            loss_metrics = [m for m in all_metrics if 'loss' in m.lower()]
            if loss_metrics:
                ax = axes[0, 0]
                for metric in loss_metrics:
                    values = [entry['metrics'].get(metric, 0) for entry in self.metrics_history]
                    ax.plot(steps, values, label=metric)
                ax.set_title('损失曲线')
                ax.set_xlabel('步数')
                ax.set_ylabel('损失值')
                ax.legend()
                ax.grid(True)
            
            # 学习率曲线
            lr_metrics = [m for m in all_metrics if 'lr' in m.lower() or 'learning_rate' in m.lower()]
            if lr_metrics:
                ax = axes[0, 1]
                for metric in lr_metrics:
                    values = [entry['metrics'].get(metric, 0) for entry in self.metrics_history]
                    ax.plot(steps, values, label=metric)
                ax.set_title('学习率曲线')
                ax.set_xlabel('步数')
                ax.set_ylabel('学习率')
                ax.legend()
                ax.grid(True)
            
            # 奖励曲线
            reward_metrics = [m for m in all_metrics if 'reward' in m.lower()]
            if reward_metrics:
                ax = axes[1, 0]
                for metric in reward_metrics:
                    values = [entry['metrics'].get(metric, 0) for entry in self.metrics_history]
                    ax.plot(steps, values, label=metric)
                ax.set_title('奖励曲线')
                ax.set_xlabel('步数')
                ax.set_ylabel('奖励值')
                ax.legend()
                ax.grid(True)
            
            # 步骤时间
            step_times = [entry['step_time'] for entry in self.metrics_history if 'step_time' in entry]
            if step_times:
                ax = axes[1, 1]
                ax.plot(steps[:len(step_times)], step_times)
                ax.set_title('步骤耗时')
                ax.set_xlabel('步数')
                ax.set_ylabel('时间(秒)')
                ax.grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            report_path = self.output_dir / "training_report.png"
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 训练报告已保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️  生成训练报告失败: {e}")

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, output_dir: str = "./output", max_checkpoints: int = 5):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点信息文件
        self.info_file = self.checkpoints_dir / "checkpoint_info.json"
        self.checkpoint_info = self._load_checkpoint_info()
    
    def _load_checkpoint_info(self) -> Dict:
        """加载检查点信息"""
        if self.info_file.exists():
            try:
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  加载检查点信息失败: {e}")
        
        return {
            "checkpoints": [],
            "best_checkpoint": None,
            "best_metric": None,
            "best_value": None
        }
    
    def _save_checkpoint_info(self):
        """保存检查点信息"""
        try:
            with open(self.info_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存检查点信息失败: {e}")
    
    def save_checkpoint(self, model, tokenizer, step: int, metrics: Dict[str, float], 
                       is_best: bool = False, metric_name: str = "loss"):
        """保存检查点"""
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        try:
            # 保存模型和分词器
            checkpoint_path.mkdir(exist_ok=True)
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # 保存训练状态
            state_dict = {
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "model_name": getattr(model.config, 'name_or_path', 'unknown')
            }
            
            with open(checkpoint_path / "training_state.json", 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
            
            # 更新检查点信息
            checkpoint_info = {
                "name": checkpoint_name,
                "step": step,
                "path": str(checkpoint_path),
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "size_mb": self._get_dir_size(checkpoint_path)
            }
            
            self.checkpoint_info["checkpoints"].append(checkpoint_info)
            
            # 检查是否是最佳检查点
            if is_best or self._is_best_checkpoint(metrics, metric_name):
                self.checkpoint_info["best_checkpoint"] = checkpoint_name
                self.checkpoint_info["best_metric"] = metric_name
                self.checkpoint_info["best_value"] = metrics.get(metric_name)
                
                # 创建最佳检查点的符号链接
                best_link = self.checkpoints_dir / "best"
                if best_link.exists():
                    if best_link.is_symlink():
                        best_link.unlink()
                    else:
                        shutil.rmtree(best_link)
                
                try:
                    best_link.symlink_to(checkpoint_path, target_is_directory=True)
                except OSError:
                    # Windows可能不支持符号链接，使用复制
                    shutil.copytree(checkpoint_path, best_link, dirs_exist_ok=True)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            # 保存更新的信息
            self._save_checkpoint_info()
            
            print(f"💾 检查点已保存: {checkpoint_name} ({checkpoint_info['size_mb']:.1f}MB)")
            if is_best:
                print(f"🏆 新的最佳检查点! {metric_name}: {metrics.get(metric_name, 'N/A')}")
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")
            return None
    
    def _is_best_checkpoint(self, metrics: Dict[str, float], metric_name: str) -> bool:
        """判断是否是最佳检查点"""
        if metric_name not in metrics:
            return False
        
        current_value = metrics[metric_name]
        best_value = self.checkpoint_info.get("best_value")
        
        if best_value is None:
            return True
        
        # 对于损失类指标，越小越好；对于准确率、奖励等，越大越好
        if "loss" in metric_name.lower() or "error" in metric_name.lower():
            return current_value < best_value
        else:
            return current_value > best_value
    
    def _get_dir_size(self, path: Path) -> float:
        """获取目录大小(MB)"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size / 1024 / 1024
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = self.checkpoint_info["checkpoints"]
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # 按步数排序，保留最新的检查点
        checkpoints.sort(key=lambda x: x["step"])
        
        # 删除最旧的检查点
        to_remove = checkpoints[:-self.max_checkpoints]
        
        for checkpoint in to_remove:
            try:
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    print(f"🗑️  删除旧检查点: {checkpoint['name']}")
            except Exception as e:
                print(f"⚠️  删除检查点失败: {e}")
        
        # 更新检查点列表
        self.checkpoint_info["checkpoints"] = checkpoints[-self.max_checkpoints:]
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点"""
        return self.checkpoint_info["checkpoints"]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        best_name = self.checkpoint_info.get("best_checkpoint")
        if best_name:
            return str(self.checkpoints_dir / best_name)
        return None
    
    def load_checkpoint(self, checkpoint_name: str = None) -> Optional[str]:
        """加载检查点"""
        if checkpoint_name is None:
            # 加载最佳检查点
            checkpoint_path = self.get_best_checkpoint()
        elif checkpoint_name == "latest":
            # 加载最新检查点
            checkpoints = self.list_checkpoints()
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x["step"])
                checkpoint_path = latest["path"]
            else:
                checkpoint_path = None
        else:
            # 加载指定检查点
            checkpoint_path = str(self.checkpoints_dir / checkpoint_name)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"📂 加载检查点: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"❌ 检查点不存在: {checkpoint_name or 'best'}")
            return None
    
    def print_summary(self):
        """打印检查点摘要"""
        checkpoints = self.list_checkpoints()
        
        print("\n📋 检查点摘要:")
        print(f"   总检查点数: {len(checkpoints)}")
        
        if checkpoints:
            total_size = sum(cp["size_mb"] for cp in checkpoints)
            print(f"   总大小: {total_size:.1f}MB")
            
            latest = max(checkpoints, key=lambda x: x["step"])
            print(f"   最新检查点: {latest['name']} (步骤 {latest['step']:,})")
            
            best_name = self.checkpoint_info.get("best_checkpoint")
            if best_name:
                best_metric = self.checkpoint_info.get("best_metric")
                best_value = self.checkpoint_info.get("best_value")
                print(f"   最佳检查点: {best_name} ({best_metric}: {best_value})")
            
            print("\n   检查点列表:")
            for cp in sorted(checkpoints, key=lambda x: x["step"]):
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in cp["metrics"].items()])
                print(f"     {cp['name']}: 步骤 {cp['step']:,} | {metrics_str} | {cp['size_mb']:.1f}MB")
        else:
            print("   无检查点")

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练监控和检查点管理工具")
    parser.add_argument("--action", choices=["monitor", "checkpoints"], default="checkpoints", help="操作类型")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    
    args = parser.parse_args()
    
    if args.action == "monitor":
        # 演示训练监控
        monitor = TrainingMonitor(args.output_dir)
        monitor.start_training(1000, "test-model")
        
        # 模拟训练步骤
        for step in range(1, 101):
            metrics = {
                "loss": 2.0 - step * 0.01,
                "learning_rate": 1e-4 * (0.99 ** step),
                "reward": step * 0.1
            }
            monitor.log_step(step, metrics)
            time.sleep(0.1)  # 模拟训练时间
        
        monitor.finish_training({"final_loss": 1.0})
        
    elif args.action == "checkpoints":
        # 显示检查点信息
        manager = CheckpointManager(args.output_dir)
        manager.print_summary()

if __name__ == "__main__":
    main()