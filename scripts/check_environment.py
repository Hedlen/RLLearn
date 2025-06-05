#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查脚本
"""

import sys
import os
import platform
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("=== Python环境检查 ===")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python版本: {version_str}")
    
    if version < (3, 8):
        print("❌ Python版本过低，建议使用3.8或更高版本")
        return False
    elif version >= (3, 11):
        print("✅ Python版本良好")
    else:
        print("✅ Python版本可用")
    
    print(f"Python路径: {sys.executable}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    return True

def check_gpu_environment():
    """检查GPU环境"""
    print("\n=== GPU环境检查 ===")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA可用，检测到 {gpu_count} 个GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 测试GPU可用性
            try:
                test_tensor = torch.randn(100, 100).cuda()
                print("✅ GPU测试通过")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU测试失败: {e}")
                return False
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
            print("   如需GPU加速，请安装CUDA版本的PyTorch")
        
        return True
        
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def check_required_packages():
    """检查必需的Python包"""
    print("\n=== 依赖包检查 ===")
    
    required_packages = [
        ('torch', 'PyTorch', '>=1.12.0'),
        ('transformers', 'Transformers', '>=4.21.0'),
        ('datasets', 'Datasets', '>=2.0.0'),
        ('numpy', 'NumPy', '>=1.20.0'),
        ('pandas', 'Pandas', '>=1.3.0'),
        ('yaml', 'PyYAML', '>=5.4.0'),
        ('tqdm', 'tqdm', '>=4.60.0'),
        ('tensorboard', 'TensorBoard', '>=2.8.0'),
        ('accelerate', 'Accelerate', '>=0.12.0'),
        ('peft', 'PEFT', '>=0.3.0')
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package_name, display_name, min_version in required_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            installed_packages.append((package_name, version))
        except ImportError:
            print(f"❌ {display_name}: 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print(f"安装命令: pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ 所有必需包已安装")
    return True

def check_system_resources():
    """检查系统资源"""
    print("\n=== 系统资源检查 ===")
    
    try:
        import psutil
        
        # 内存检查
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
        
        print(f"系统内存: {memory_gb:.1f}GB (可用: {available_gb:.1f}GB)")
        
        if memory_gb < 8:
            print("⚠️  系统内存较少，建议至少8GB")
        elif memory_gb >= 16:
            print("✅ 系统内存充足")
        else:
            print("✅ 系统内存可用")
        
        # CPU检查
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        print(f"CPU核心: {cpu_count} 物理核心, {cpu_count_logical} 逻辑核心")
        
        # 磁盘空间检查
        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1024**3
        total_gb = disk.total / 1024**3
        
        print(f"磁盘空间: {free_gb:.1f}GB 可用 / {total_gb:.1f}GB 总计")
        
        if free_gb < 10:
            print("⚠️  磁盘空间不足，建议至少保留10GB")
        else:
            print("✅ 磁盘空间充足")
        
        return True
        
    except ImportError:
        print("⚠️  psutil未安装，无法检查系统资源")
        print("   安装命令: pip install psutil")
        return True  # 不是必需的，所以返回True
    except Exception as e:
        print(f"❌ 系统资源检查失败: {e}")
        return False

def check_huggingface_access():
    """检查HuggingFace访问"""
    print("\n=== HuggingFace访问检查 ===")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试模型访问
        test_model = "Qwen/Qwen2.5-3B-Instruct"
        print(f"测试模型访问: {test_model}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(test_model)
            print("✅ HuggingFace模型访问正常")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "timeout" in error_msg:
                print("❌ 网络连接问题，无法访问HuggingFace")
                print("   建议:")
                print("   1. 检查网络连接")
                print("   2. 使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
                print("   3. 使用代理或VPN")
            else:
                print(f"❌ HuggingFace访问失败: {e}")
            return False
            
    except ImportError:
        print("❌ transformers未安装")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n=== 项目结构检查 ===")
    
    required_dirs = ['src', 'data', 'output', 'logs']
    required_files = ['config.yaml', 'requirements.txt']
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ 目录存在: {dir_name}/")
        else:
            print(f"📁 创建目录: {dir_name}/")
            dir_path.mkdir(exist_ok=True)
    
    # 检查文件
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✅ 文件存在: {file_name}")
        else:
            print(f"⚠️  文件不存在: {file_name}")
    
    return True

def generate_environment_report():
    """生成环境报告"""
    print("\n=== 生成环境报告 ===")
    
    report = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': f"{platform.system()} {platform.release()}",
        'python_path': sys.executable
    }
    
    # GPU信息
    try:
        import torch
        report['pytorch_version'] = torch.__version__
        report['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report['gpu_count'] = torch.cuda.device_count()
            report['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        report['pytorch_version'] = 'Not installed'
        report['cuda_available'] = False
    
    # 系统资源
    try:
        import psutil
        memory = psutil.virtual_memory()
        report['total_memory_gb'] = round(memory.total / 1024**3, 1)
        report['available_memory_gb'] = round(memory.available / 1024**3, 1)
        report['cpu_count'] = psutil.cpu_count()
        
        disk = psutil.disk_usage('.')
        report['free_disk_gb'] = round(disk.free / 1024**3, 1)
    except ImportError:
        pass
    
    # 保存报告
    import json
    report_file = Path('logs/environment_report.json')
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 环境报告已保存: {report_file}")
    return report_file

def main():
    """主函数"""
    print("🔍 开始环境检查...\n")
    
    checks = [
        check_python_version,
        check_required_packages,
        check_gpu_environment,
        check_system_resources,
        check_huggingface_access,
        check_project_structure
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append(False)
    
    # 生成报告
    generate_environment_report()
    
    # 总结
    print("\n" + "="*50)
    print("🎯 环境检查总结")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("✅ 所有检查通过，环境配置良好！")
        print("🚀 可以开始训练了")
        return True
    else:
        print(f"⚠️  {total - passed} 项检查未通过")
        print("请根据上述提示解决问题后重新检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)