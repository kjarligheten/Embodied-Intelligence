# FastSAM 替代方案部署指南

## 概述

由于SAM3需要申请访问权限且可能被拒绝，我们推荐使用 **FastSAM** 作为替代方案。

**FastSAM 优势：**
- ✅ 完全开源，无需申请权限
- ✅ 性能与SAM相当
- ✅ 速度快50倍
- ✅ 模型可直接下载
- ✅ Apache 2.0许可证
- ✅ 安装简单

## 步骤 1: 安装 FastSAM

### 1.1 安装 FastSAM 包
```powershell
pin install segment-anything-fast
```

### 1.2 安装 CLIP（用于文本提示）
```powershell
pip install git+https://github.com/openai/CLIP.git
```

## 步骤 2: 下载模型权重

### 2.1 创建模型目录
```powershell
mkdir models
mkdir models\FastSAM
```

### 2.2 下载 FastSAM 模型

**选项 A: Google Drive（推荐）**
- **FastSAM-x** (默认，性能最好): https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing
- **FastSAM-s** (小型，速度快): https://drive.google.com/file/d/10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV/view?usp=sharing

**选项 B: 百度云（国内推荐）**
- 提取码: 0000
- 链接: https://pan.baidu.com/s/18KzBmOTENjByoWWR17zdiQ?pwd=0000

### 2.3 手动下载步骤
1. 点击上述链接
2. 下载模型文件
3. 将文件保存到 `models\FastSAM\` 目录
4. 重命名为 `FastSAM-x.pt` 或 `FastSAM-s.pt`

## 步骤 3: 更新 sam3_server.py 使用 FastSAM

我将帮你更新服务器代码以使用FastSAM替代SAM3。

### 3.1 FastSAM 与 SAM3 的接口差异

FastSAM 使用不同的API，需要适配：
- FastSAM 使用 YOLOv8 架构
- 支持文本提示分割
- 输出格式与SAM3类似

### 3.2 代码修改要点

主要修改：
1. 导入 FastSAM 相关库
2. 修改模型加载逻辑
3. 适配推理接口
4. 保持输出格式兼容

## 步骤 4: 测试 FastSAM

### 4.1 重启服务器
```powershell
# 停止当前服务器
taskkill /F /IM python.exe

# 启动新的服务器
python sam3_server.py
```

### 4.2 测试健康检查
```powershell
python -c "import requests; print(requests.get('http://127.0.0.1:8766/health').json())"
```

### 4.3 运行策略测试
```powershell
python test_improved_policy.py
```

## FastSAM 模型对比

| 模型 | 大小 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|----------|
| FastSAM-x | ~140MB | 快 | 高 | 生产环境，平衡性能 |
| FastSAM-s | ~40MB | 最快 | 中等 | 开发测试，资源受限 |

**推荐使用 FastSAM-x** 以获得更好的分割质量。

## 性能对比

### FastSAM vs SAM3

| 指标 | SAM3 | FastSAM |
|------|------|---------|
| 推理速度 | 基准 | 50× 更快 |
| 模型大小 | ~2.4GB | ~140MB |
| 分割质量 | 最高 | 相当 |
| 访问权限 | 需要申请 | 无需申请 |
| 许可证 | Meta | Apache 2.0 |

## 故障排除

### 问题 1: 模型下载失败
**解决**: 
- 使用百度云链接（国内更快）
- 检查网络连接
- 尝试使用下载工具

### 问题 2: CLIP 安装失败
**解决**:
```powershell
pip install openai-clip
```

### 问题 3: 推理速度慢
**解决**:
- 使用 FastSAM-s 模型
- 如果有GPU，确保CUDA正确配置
- 减少输入图像分辨率

### 问题 4: 分割质量不如预期
**解决**:
- 使用 FastSAM-x 模型
- 调整文本提示
- 增加置信度阈值

## 下一步

完成FastSAM部署后：
1. 我将更新 sam3_server.py 代码
2. 测试FastSAM服务器
3. 运行策略测试验证集成
4. 对比FastSAM与模拟分割的效果

## 参考资源

- FastSAM GitHub: https://github.com/CASIA-IVA-Lab/FastSAM
- FastSAM 论文: https://arxiv.org/pdf/2306.12156.pdf
- HuggingFace Demo: https://huggingface.co/spaces/An-619/FastSAM
