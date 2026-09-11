# Model Services Setup Guide

This guide explains how to configure the VLM and SAM3 services required for the mujoco-grasp-assignment.

## Required Environment Variables

### VLM (Visual Language Model) Service

The VLM service is used for instruction understanding and target grounding.

**Environment Variables:**
- `GRASPBENCH_VLM_BASE_URL`: Base URL of the OpenAI-compatible VLM API (e.g., `https://api.openai.com/v1`)
- `GRASPBENCH_VLM_API_KEY`: API key for authentication
- `GRASPBENCH_VLM_MODEL`: Model name (default: `qwen3-vl-flash`)
- `GRASPBENCH_VLM_TIMEOUT_S`: Request timeout in seconds (default: `45`)

**Supported VLM Services:**
- OpenAI GPT-4 Vision
- Qwen3-VL (recommended for this assignment)
- DeepSeek-VL
- Any OpenAI-compatible vision-language model

**Example Setup (PowerShell):**
```powershell
$env:GRASPBENCH_VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:GRASPBENCH_VLM_API_KEY = "your-api-key-here"
$env:GRASPBENCH_VLM_MODEL = "qwen-vl-max"
```

### SAM3 (Segment Anything Model) Service

The SAM3 service is used for object segmentation from RGB images.

**Environment Variables:**
- `GRASPBENCH_SAM3_URL`: URL of the SAM3 inference service (default: `http://127.0.0.1:8765/infer`)

**SAM3 Service Setup:**

You need to run a SAM3 inference server. One option is to use the official SAM implementation:

```bash
# Install SAM3 dependencies
pip install segment-anything fastapi uvicorn pillow

# Run SAM3 server (example)
uvicorn sam3_server:app --host 0.0.0.0 --port 8765
```

**Example Setup (PowerShell):**
```powershell
$env:GRASPBENCH_SAM3_URL = "http://127.0.0.1:8765/infer"
```

## Windows Compatibility Variables

For Windows systems with Chinese characters in paths:

```powershell
$env:MUJOCO_GL = "glfw"
$env:GRASPBENCH_ASSET_DIR = "C:/mujoco_test/assets"
```

## Complete Setup Example (PowerShell)

```powershell
# Set MuJoCo and asset paths
$env:MUJOCO_GL = "glfw"
$env:GRASPBENCH_ASSET_DIR = "C:/mujoco_test/assets"

# Set VLM configuration
$env:GRASPBENCH_VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:GRASPBENCH_VLM_API_KEY = "your-api-key-here"
$env:GRASPBENCH_VLM_MODEL = "qwen-vl-max"

# Set SAM3 configuration
$env:GRASPBENCH_SAM3_URL = "http://127.0.0.1:8765/infer"

# Run tests
python -m pytest tests/ -v
```

## Testing Without Model Services

If you don't have model services configured yet, you can still test:

1. **Basic functionality tests** (no model services required):
   ```powershell
   python test_functionality.py
   ```

2. **Unit tests** (mock model services):
   ```powershell
   python -m pytest tests/test_environment.py tests/test_ik_and_camera.py -v
   ```

## Troubleshooting

### VLM Connection Issues
- Check that the base URL is correct and accessible
- Verify the API key is valid
- Ensure the model name is supported by the service

### SAM3 Connection Issues
- Ensure the SAM3 server is running on the specified port
- Check firewall settings
- Verify the service accepts POST requests with image data

### Path Issues on Windows
- Always set `GRASPBENCH_ASSET_DIR` to a path without Chinese characters
- Use forward slashes in paths
- Set `MUJOCO_GL=gw` for OpenGL context
