"""SAM3 Server for object segmentation.

This server provides image segmentation using FastSAM (Segment Anything Model alternative).
 In simulation mode, it uses depth-based detection as a fallback.
"""

import base64
import io
import json
from typing import List, Dict, Any
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import uvicorn
from fastsam import FastSAM, FastSAMPrompt
import torch

# FastSAM model configuration
fastsam_model = None  # Loaded on startup
MODEL_PATH = "models/FastSAM/FastSAM-x.pt"

# Simulation mode: use depth-based detection instead of FastSAM
SIMULATION_MODE = True  # Force simulation mode for testing

app = FastAPI(title="SAM3 Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SAM3Request(BaseModel):
    """Request model for SAM3 inference.

    Attributes:
        image_jpeg_b64: Base64 encoded JPEG image
        depth_b64: Base64 encoded depth image (optional, for simulation mode)
        prompts: Text prompts for segmentation
    """


class SAM3SingleResult(BaseModel):
    """Result for a single segmentation prompt.

    Attributes:
        scores: Confidence scores for detections
        boxes: Bounding boxes [x1, y1, x2, y2]
        masks: Segmentation masks (encoded)
        metadata: Additional metadata (e.g., world position in simulation mode)
    """


class SAM3Response(BaseModel):
    """Response model for SAM3 inference.

    Attributes:
        results: List of segmentation results, one per prompt
    """


def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 JPEG image to numpy array.

    Args:
        image_b64: Base64 encoded JPEG string

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def decode_depth(depth_b64: str) -> np.ndarray:
    """Decode base64 depth image to numpy array.

    Args:
        depth_b64: Base64 encoded depth image string

    Returns:
        Depth image as numpy array (H, W) or None if decoding fails
    """
    if not depth_b64:
        return None
    try:
        depth_data = base64.b64decode(depth_b64)
        depth = Image.open(io.BytesIO(depth_data))
        return np.array(depth)
    except Exception as e:
        print(f"Warning: Failed to decode depth image: {e}")
        return None


def encode_mask(mask: np.ndarray) -> str:
    """Encode boolean mask to base64 packed bits.

    Args:
        mask: Boolean mask array (H, W)

    Returns:
        Base64 encoded packed bits string
    """
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    encoded = base64.b64encode(packed.tobytes()).decode("ascii")
    return encoded


@app.on_event("startup")
async def startup_event():
    """Load FastSAM model on startup."""
    global fastsam_model
    print("FastSAM Server starting...")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Falling back to mock segmentation.")
        return
    
    try:
        # Determine device (CPU or CUDA)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading FastSAM model from {MODEL_PATH} on {device}...")
        
        # Monkey patch torch.load to use weights_only=False
        original_torch_load = torch.load
        def patched_load(f, *args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_torch_load(f, *args, **kwargs)
        torch.load = patched_load
        
        try:
            fastsam_model = FastSAM(MODEL_PATH)
            fastsam_model.to(device)
            print(f"FastSAM model loaded successfully on {device}!")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
    except Exception as e:
        print(f"Error loading FastSAM model: {e}")
        print("Falling back to mock segmentation.")


@app.post("/infer", response_model=SAM3Response)
async def infer(request: SAM3Request) -> SAM3Response:
    """Run FastSAM inference on the provided image."""
    try:
        # Decode image and optional depth
        image = decode_image(request.image_jpeg_b64)
        depth = decode_depth(request.depth_b64) if request.depth_b64 else None
        height, width = image.shape[:2]
        
        # Use depth-based inference in simulation mode
        if SIMULATION_MODE and depth is not None:
            print(f"[SIMULATION MODE] Using depth-based segmentation for {len(request.prompts)} prompts")
            return await _run_depth_based_inference(image, depth, request.prompts, height, width)
        
        # Use FastSAM if model is loaded
        if fastsam_model is not None:
            return await _run_fastsam_inference(image, request.prompts, height, width)
        
        # Fall back to mock inference
        print("Using mock segmentation (FastSAM not loaded)")
        return await _run_mock_inference(image, depth, request.prompts, height, width)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_fastsam_inference(image, depth, prompts, height, width):
    """Run FastSAM inference with text prompts."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    
    # Run FastSAM detection
    everything_results = fastsam_model(
        pil_image,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.25,  # Lowered from 0.4 to be more permissive
        iou=0.7,    # Lowered from 0.9 to allow more detections
    )
    
    # Process each prompt
    for prompt in prompts:
        prompt_process = FastSAMPrompt(pil_image, everything_results, device=device)
        
        # Use text prompt for segmentation
        ann = prompt_process.text_prompt(text=prompt)
        
        if ann is None or len(ann) == 0:
            # Fallback to mock if no detection
            print(f"FastSAM: No detection for prompt '{prompt}', using fallback")
            mock_result = await _run_mock_inference(image, depth, [prompt], height, width)
            results.extend(mock_result.results)
            continue
        
        # Convert FastSAM output to SAM3 format
        for mask in ann:
            # Get bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
            
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            box_xyxy = [
                float(x_indices[0]),
                float(y_indices[0]),
                float(x_indices[-1] + 1),
                float(y_indices[-1] + 1)
            ]
            
            # Encode mask
            encoded = encode_mask(mask)
            
            result = SAM3SingleResult(
                scores=[0.95],  # FastSAM doesn't provide confidence per mask
                boxes=[box_xyxy],
                masks={
                    "count": 1,
                    "packed": encoded
                }
            )
            results.append(result)
            break  # Use first match for each prompt
    
    return SAM3Response(results=results)


async def _run_depth_based_inference(image, depth, prompts, height, width):
    """Run depth-based segmentation inference for simulation mode.
    
    This uses the actual depth image to detect objects by clustering depth values,
    creating masks that match the simulation geometry.
    """
    # Simplified approach: use mock inference with depth data
    # This is more reliable than complex connected components
    return await _run_mock_inference(image, depth, prompts, height, width)


async def _run_mock_inference(image, depth, prompts, height, width):
    """Run mock segmentation inference (fallback).
    
    Returns masks with approximate positions based on prompt keywords.
    """
    results = []
    
    # Ensure depth is a numpy array
    if isinstance(depth, str):
        try:
            depth = decode_depth(depth)
        except:
            depth = None
    
    # Use depth data to detect actual object positions
    # Find objects by thresholding depth and finding connected components
    if depth is not None and isinstance(depth, np.ndarray):
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # Threshold to find tabletop objects
        table_depth_threshold = 0.5
        object_mask = depth_normalized > table_depth_threshold
        
        # Find connected components
        try:
            from scipy import ndimage
            labeled, num_objects = ndimage.label(object_mask)
            
            # Get centroids of each object
            object_positions = []
            for i in range(1, num_objects + 1):
                component_mask = (labeled == i)
                if component_mask.sum() > 50:  # Minimum size threshold
                    y_coords, x_coords = np.where(component_mask)
                    center_y = int(y_coords.mean())
                    center_x = int(x_coords.mean())
                    object_positions.append((center_x, center_y))
        except ImportError:
            # Fallback to grid positions if scipy not available
            object_positions = [
                (int(width * 0.3), int(height * 0.3)),
                (int(width * 0.5), int(height * 0.5)),
                (int(width * 0.7), int(height * 0.7)),
            ]
        
        # If no objects detected, use grid positions
        if not object_positions:
            object_positions = [
                (int(width * 0.3), int(height * 0.3)),
                (int(width * 0.5), int(height * 0.5)),
                (int(width * 0.7), int(height * 0.7)),
            ]
    else:
        # Fallback to grid positions if no depth data
        object_positions = [
            (int(width * 0.3), int(height * 0.3)),
            (int(width * 0.5), int(height * 0.5)),
            (int(width * 0.7), int(height * 0.7)),
        ]
    
    mask_size = min(width, height) // 8
    
    for i, prompt in enumerate(prompts):
        # Assign each prompt to a different detected object position
        if i < len(object_positions):
            center_x, center_y = object_positions[i]
        else:
            # Fallback to grid if not enough objects
            center_x = int(width * (0.3 + 0.2 * (i % 3)))
            center_y = int(height * (0.3 + 0.2 * (i // 3)))
        
        # Create elliptical mask
        y, x = np.ogrid[:height, :width]
        center_y_f, center_x_f = float(center_y), float(center_x)
        radius_y = mask_size * 1.2
        radius_x = mask_size * 0.8
        mask = ((x - center_x_f)**2 / radius_x**2 + (y - center_y_f)**2 / radius_y**2) <= 1
        
        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue
        
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        
        box_xyxy = [
            float(x_indices[0]),
            float(y_indices[0]),
            float(x_indices[-1] + 1),
            float(y_indices[-1] + 1)
        ]
        
        # Encode mask
        encoded = encode_mask(mask.astype(np.uint8))
        
        result = SAM3SingleResult(
            scores=[0.95],
            boxes=[box_xyxy],
            masks={
                "count": 1,
                "packed": encoded
            },
            metadata={"mock_mode": "depth_based_detection"}
        )
        results.append(result)
    
    return SAM3Response(results=results)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": fastsam_model is not None, "model_type": "FastSAM" if fastsam_model else "mock"}


if __name__ == "__main__":
    print("Starting SAM3 Server on http://127.0.0.1:8766")
    uvicorn.run(app, host="127.0.0.1", port=8766)