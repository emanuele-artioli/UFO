"""UFO: Unified Framework for Salient Object Detection in Video.

This module provides a clean functional API for generating saliency masks from video frames.
"""
import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import torch
from torchvision import transforms
from .model_video import build_model
import numpy as np
import importlib


# Global model cache to avoid reloading weights on every call
_model_cache = {}


def _ensure_model_weights(model_name: str = 'model_best.pth') -> Optional[Path]:
    """Ensure model weights exist, downloading if necessary.
    
    Args:
        model_name: Name of the model weights file to look for.
        
    Returns:
        Path to the weights file, or None if not found.
    """
    # Resolve installed package weights directory
    try:
        import ufo as _ufo_pkg
        pkg_weights_dir = Path(_ufo_pkg.__file__).parent / 'weights'
        pkg_weights_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback to local package path
        pkg_weights_dir = Path(__file__).resolve().parent / 'weights'
        pkg_weights_dir.mkdir(parents=True, exist_ok=True)

    # Candidate files to accept
    candidates = [
        pkg_weights_dir / model_name,
        pkg_weights_dir / 'video_best.pth',
        pkg_weights_dir / 'ufo_weights.pth',
        pkg_weights_dir / 'video_weights.pth',
        pkg_weights_dir / 'weights.pth',
    ]

    # If any candidate exists, return it
    for c in candidates:
        if c.exists():
            return c

    # Otherwise try to run the downloader
    try:
        downloader = importlib.import_module('.download_ufo_weights', package=__package__)
        if hasattr(downloader, 'main'):
            downloaded = downloader.main()
            if downloaded is not None:
                return Path(downloaded)
    except Exception:
        pass

    # Re-check candidates and return first that exists
    for c in candidates:
        if c.exists():
            return c

    return None


def _get_model(device: torch.device, model_path: Optional[str] = None) -> torch.nn.Module:
    """Get or create a cached model instance.
    
    Args:
        device: Torch device to use.
        model_path: Optional path to model weights. If None, uses default packaged weights.
        
    Returns:
        Loaded and cached model ready for inference.
    """
    cache_key = (str(device), model_path)
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Resolve model path
    if model_path is None or not os.path.exists(model_path):
        pkg_model = _ensure_model_weights()
        if pkg_model is not None and pkg_model.exists():
            model_path = str(pkg_model)
        elif model_path is None:
            raise FileNotFoundError("Could not find UFO model weights. Run `python -m ufo.download_ufo_weights` first.")
    
    # Build and load model
    net = build_model(device).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    net = net.module.to(device)
    
    _model_cache[cache_key] = net
    return net


def segment_frames(
    frames: np.ndarray,
    device: Union[str, torch.device] = 'cuda:0',
    model_path: Optional[str] = None,
    group_size: int = 5,
    img_size: int = 224
) -> np.ndarray:
    """Generate saliency masks for a sequence of video frames.
    
    This is the main entry point for the UFO segmentation model. It takes a numpy
    array of RGB frames and returns corresponding binary saliency masks.
    
    Args:
        frames: Input frames as a numpy array of shape (N, H, W, 3) where N is the
            number of frames, H is height, W is width, and channels are in RGB order.
            Values should be in range [0, 255] as uint8.
        device: Torch device to use for inference (e.g., 'cuda:0', 'cpu').
        model_path: Optional path to model weights. If None, uses packaged weights.
        group_size: Number of frames to process together. Default 5.
        img_size: Internal processing resolution. Default 224.
        
    Returns:
        Numpy array of shape (N, H, W) containing saliency masks with values in [0, 1].
        Each mask has the same spatial dimensions as the input frames.
        
    Example:
        >>> import numpy as np
        >>> from ufo import segment_frames
        >>> # Assume frames is a numpy array of shape (10, 480, 640, 3)
        >>> masks = segment_frames(frames, device='cuda:0')
        >>> print(masks.shape)  # (10, 480, 640)
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Validate input
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (N, H, W, 3), got shape {frames.shape}")
    
    num_frames, orig_h, orig_w, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {channels}")
    
    # Get model
    net = _get_model(device, model_path)
    
    # Setup transforms
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert frames to tensor
    frame_tensors = torch.zeros(num_frames, 3, img_size, img_size)
    for i in range(num_frames):
        img = Image.fromarray(frames[i])
        frame_tensors[i] = img_transform(img)
    
    # Process frames in groups
    masks_tensor = torch.zeros(num_frames, img_size, img_size)
    divided = num_frames // group_size
    rested = num_frames % group_size
    
    with torch.no_grad():
        # Process full groups
        for k in range(divided):
            start_idx = k * group_size
            end_idx = (k + 1) * group_size
            group_rgb = frame_tensors[start_idx:end_idx].to(device)
            _, pred_mask = net(group_rgb)
            masks_tensor[start_idx:end_idx] = pred_mask.cpu()
        
        # Process remaining frames
        if rested != 0:
            group_rgb_tmp_l = frame_tensors[-rested:]
            group_rgb_tmp_r = frame_tensors[:group_size - rested]
            group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
            group_rgb = group_rgb.to(device)
            _, pred_mask = net(group_rgb)
            masks_tensor[(divided * group_size):] = pred_mask[:rested].cpu()
    
    # Resize masks back to original resolution
    masks = np.zeros((num_frames, orig_h, orig_w), dtype=np.float32)
    for i in range(num_frames):
        mask_img = Image.fromarray((masks_tensor[i].numpy() * 255).astype(np.uint8))
        mask_img = mask_img.resize((orig_w, orig_h), Image.BILINEAR)
        masks[i] = np.array(mask_img, dtype=np.float32) / 255.0
    
    return masks


def clear_model_cache():
    """Clear the cached model to free GPU memory."""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache()