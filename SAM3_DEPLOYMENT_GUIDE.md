# SAM3 真实模型部署指南

## 概述

本指南帮助你在本地部署真实的SAM3模型，以替代当前的模拟分割。

## 前置条件

✅ 已安装依赖：
- `segment-anything` (已安装)
- `huggingface_hub` (已安装)
- `fastapi` (已安装)
- `uvicorn` (已安装)
- `pillow` (已安装)

## 步骤 1: Hugging Face 认证

### 1.1 创建 Hugging Face 账户
如果你还没有账户，请访问 https://huggingface.co/ 注册。

### 1.2 生成访问令牌
1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Read" 权限
4. 复制生成的令牌

### 1.3 登录 Hugging Face CLI
在 PowerShell 中运行：
```powershell
hf auth login
```
粘贴你的令牌并按回车。

## 步骤 2: 申请 SAM3 访问权限

### 2.1 访问 SAM3 模型页面
访问以下页面之一：
- SAM3: https://huggingface.co/facebook/sam3
- SAM3.1 (推荐): https://huggingface.co/facebook/sam3.1

### 2.2 申请访问
1. 点击页面上的 "Access request" 或 "Request access"
2. 填写申请表单（通常需要说明用途）
3. 等待批准（通常几分钟到几小时）

## 步骤 3: 下载模型权重

一旦获得访问权限，使用以下命令下载模型：

### 3.1 下载 SAM3.1 模型（推荐）
```powershell
# 创建模型目录
mkdir models

# 下载 SAM3.1 模型
hf download facebook/sam3.1 --local-dir models/sam3.1 --local-dir-use-symlinks False
```

### 3.2 或者下载 SAM3 模型
```powershell
hf download facebook/sam3 --local-dir models/sam3 --local-dir-use-symlinks False
```

### 3.3 验证下载
检查模型文件是否下载成功：
```powershell
dir models\sam3.1
```

应该看到类似以下文件：
- `sam3.1_huge_vit_h_4bit.safetensors`
- `sam3.1_huge_vit_h.safetensors`
- `sam3.1_large_vit_l.safetensors`
- `sam3.1_small_vit_b.safetensors`

## 步骤 4: 更新 sam3_server.py

sam3_server.py 需要更新以加载真实的SAM3模型。文件已经包含了加载代码的注释，只需要取消注释并配置模型路径。

### 4.1 设置模型路径
在 sam3_server.py 中，修改模型路径：
```python
# 在文件顶部添加模型路径配置
SAM3_MODEL_PATH = "models/sam3.1/sam3.1_large_vit_l.safetensors"
```

### 4.2 启用真实模型加载
取消注释 `startup_event` 函数中的模型加载代码。

## 步骤 5: 测试真实模型

### 5.1 重启 SAM3 服务器
```powershell
# 停止当前服务器（如果运行中）
taskkill /F /IM python.exe

# 启动新的服务器
python sam3_server.py
```

### 5.2 测试服务器
```powershell
python -c "import requests; print(requests.get('http://127.0.0.1:8766/health').json())"
```

应该看到：
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 5.3 运行策略测试
```powershell
python test_improved_policy.py
```

## 故障排除

### 问题 1: 访问被拒绝
**错误**: `Access to model facebook/sam3 is restricted`
**解决**: 确保你已在 Hugging Face 上申请并获得访问权限，并且已登录。

### 问题 2: 模型加载失败
**错误**: `FileNotFoundError` 或模型加载错误
**解决**: 
- 检查模型路径是否正确
- 确保模型文件完整下载
- 检查磁盘空间是否足够

### 问题 3: 内存不足
**错误**: CUDA out of memory 或内存错误
**解决**: 
- 使用较小的模型（如 `sam3.1_small_vit_b.safetensors`）
- 减少批处理大小
- 如果使用CPU，确保有足够内存

### 问题 4: 推理速度慢
**解决**: 
- 使用较小的模型
- 如果有GPU，确保CUDA正确配置
- 考虑使用量化模型（4-bit）

## 模型选择建议

| 模型 | 大小 | 精度 | 速度 | 推荐场景 |
|------|------|------|------|----------|
| sam3.1_huge_vit_h | ~2.4GB | 最高 | 最慢 | 生产环境，最佳质量 |
| sam3.1_large_vit_l | ~1.2GB | 高 | 中等 | 平衡性能和质量 |
| sam3.1_small_vit_b | ~0.3GB | 中等 | 快 | 开发测试，资源受限 |

对于本项目，推荐使用 `sam3.1_large_vit_l` 以平衡性能和质量。

## 下一步

完成SAM3部署后：
1. 运行 `test_improved_policy.py` 测试策略集成
2. 运行完整的评估脚本测试所有任务
3. 如果成功，可以移除模拟分割代码
