
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch.nn.functional as F
import cv2
import requests



def rescale(X):
    X = 2 * X - 1.0
    return X


def inverse_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)


def imageresize2tensor(path, image_size):
    img = Image.open(path)
    convert = transforms.Compose(
        [transforms.Resize(image_size, interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)


def image2tensor(path):
    img = Image.open(path)
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)


def calculate_l2_given_paths(path1, path2):
    file_name_old = os.listdir(path1)
    total = 0
    file_name = []
    for filename in file_name_old:
        if "fake" in str(filename):
            file_name.append(filename)

    for name in file_name:
        s = imageresize2tensor(os.path.join(path1, name.replace("fake", "real")), 256)
        name_i = name.split('.')[0]
        name = name_i + '.png'
        t = imageresize2tensor(os.path.join(path2, name), 256)
        l2_i = torch.norm(s - t, p=2)
        total += l2_i
    return total / len(file_name)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def cosine_similarity(X, Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X) * norm(Y)  # (B,C,H*W)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity


def sam_encode(sam_model, image, image_generated,device):
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resampled_image = sam_transform.apply_image(image)
        # resized_shapes.append(resize_img.shape[:2])
        resampled_image_tensor = torch.as_tensor(resampled_image.transpose(2, 0, 1)).to(device)
        # model input: (1, 3, 1024, 1024)
        resampled_image = sam_model.preprocess(resampled_image_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        assert resampled_image.shape == (1, 3, sam_model.image_encoder.img_size,
                                     sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        resampled_image_generated = sam_transform.apply_image(image_generated)
        resampled_image_generated_tensor = torch.as_tensor(resampled_image_generated.transpose(2, 0, 1)).to(device)
        resampled_image_generated = sam_model.preprocess(resampled_image_generated_tensor[None, :, :, :])
        assert resampled_image_generated.shape == (1, 3, sam_model.image_encoder.img_size,
                                            sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        # input_imgs.append(input_image.cpu().numpy()[0])
        with torch.no_grad():
            embedding = sam_model.image_encoder(resampled_image)
            embedding_generated = sam_model.image_encoder(resampled_image_generated)
            samscore = cosine_similarity(embedding, embedding_generated)
        return samscore


def sam_encode_from_torch(sam_model, image, image_generated, device, requires_grad=False):
    """
    Compute SAM-based similarity score between two images.
    
    Args:
        sam_model: The SAM model
        image: Source image tensor
        image_generated: Generated image tensor  
        device: Device to run computation on
        requires_grad: If True, allow gradients to flow through the computation.
                      If False, wrap in torch.no_grad() for evaluation only.
    
    Returns:
        SAM similarity score (higher is better)
    """
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resampled_image = sam_transform.apply_image_torch(image).to(device)
        resampled_image = sam_model.preprocess(resampled_image)

        resampled_image_generated = sam_transform.apply_image_torch(image_generated).to(device)
        resampled_image_generated = sam_model.preprocess(resampled_image_generated)

        assert resampled_image.shape == (resampled_image.shape[0], 3, sam_model.image_encoder.img_size,sam_model.image_encoder.img_size), 'input image should be resized to 3*1024*1024'
        assert resampled_image_generated.shape == (resampled_image.shape[0], 3, sam_model.image_encoder.img_size,
                                            sam_model.image_encoder.img_size), 'input image should be resized to 3*1024*1024'

        if requires_grad:
            # Allow gradients to flow for training loss
            embedding = sam_model.image_encoder(resampled_image)
            embedding_generated = sam_model.image_encoder(resampled_image_generated)
            samscore = cosine_similarity(embedding, embedding_generated)
        else:
            # Use no_grad for evaluation/logging only
            with torch.no_grad():
                embedding = sam_model.image_encoder(resampled_image)
                embedding_generated = sam_model.image_encoder(resampled_image_generated)
                samscore = cosine_similarity(embedding, embedding_generated)
        return samscore

def download_model(url,model_name,destination):

    chunk_size = 8192  # Size of each chunk in bytes

    response = requests.get(url+model_name, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print("Weights downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)


import os
import torch
import torch.nn as nn
from typing import Optional, Union

# assume these are imported from your SAM libs
# from segment_anything import sam_model_registry
# from mobile_sam import mobile_sam_model_registry

class SAMScore(nn.Module):
    """
    A wrapper around SAM/MedSAM to compute a perceptual-style score.

    Args:
        debug (bool): If True, force CPU-safe loading and extra logging.
        model_type (str): 'vit_b', 'vit_l', 'vit_h', or 'vit_t' (mobile).
        model_weight_path (str): Path to the .pth/.pt weights file.
        version (str): Kept for API compatibility; not used internally.

    Notes:
        - Power users can pass `debug=True` on clusters without GPUs.
        - We avoid `checkpoint=torch.load(...dict...)` — we either give SAM a path
          or we load and `load_state_dict` ourselves.
    """

    def __init__(
        self,
        debug: bool,
        model_type: str = "vit_b",
        model_weight_path: Union[str, os.PathLike] = "weights/medsam_vit_b.pth",
        version: str = "1.0",
    ):
        super().__init__()
        self.version = version
        self.model_type = model_type.lower().strip()

        # Decide device early
        self.device = torch.device("cuda" if torch.cuda.is_available() and not debug else "cpu")

        # Validate path
        model_weight_path = os.fspath(model_weight_path)
        if not os.path.exists(model_weight_path):
            raise FileNotFoundError(f"[SAMScore] Weights not found: {model_weight_path}")

        # Build the bare model first (no checkpoint dict passed)
        if self.model_type == "vit_t":
            # Mobile SAM registry may accept `checkpoint=path`, but we’ll load explicitly for consistency
            self.sam = mobile_sam_model_registry[self.model_type]()  # type: ignore
        else:
            self.sam = sam_model_registry[self.model_type]()  # type: ignore

        # Load checkpoint safely
        ckpt = torch.load(model_weight_path, map_location="cpu")  # always load to CPU first

        # Unwrap common nested formats
        if isinstance(ckpt, dict):
            for k in ("model", "state_dict", "weights", "module"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break

        # Load weights with tolerance for minor key diffs
        missing, unexpected = self.sam.load_state_dict(ckpt, strict=False)
        if debug:
            if missing:
                print(f"[SAMScore] Missing keys ({len(missing)}): {list(missing)[:10]}{' ...' if len(missing)>10 else ''}")
            if unexpected:
                print(f"[SAMScore] Unexpected keys ({len(unexpected)}): {list(unexpected)[:10]}{' ...' if len(unexpected)>10 else ''}")

        # Finalize
        self.sam.to(self.device)
        self.sam.eval()
        # Freeze params (usually desired for scoring)
        for p in self.sam.parameters():
            p.requires_grad = False



    def evaluation_from_path(self, source_image_path=None,  generated_image_path=None):
        source_cv2 = cv2.imread(source_image_path)
        generated_cv2 = cv2.imread(generated_image_path)
        samscore = sam_encode(self.sam, source_cv2, generated_cv2,device = self.device)
        return samscore


    def evaluation_from_torch(self, source, generated, requires_grad=False):
        """
        Compute SAM score from PyTorch tensors.
        
        Args:
            source: Source image tensor
            generated: Generated image tensor
            requires_grad: If True, allow gradients to flow (for training loss).
                          If False, compute in no_grad mode (for logging/evaluation).
        
        Returns:
            SAM similarity score
        """
        samscore = sam_encode_from_torch(self.sam, source, generated, device=self.device, requires_grad=requires_grad)
        return samscore

    def _get_mask_generator(self):
        """Lazy initialization of mask generator"""
        if not hasattr(self, '_mask_generator'):
            self._mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=100,
            )
        return self._mask_generator

    def _tensor_to_numpy_image(self, tensor):
        """Convert tensor [B, C, H, W] in [0, 255] to numpy [H, W, C] uint8"""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image in batch
        # tensor is [C, H, W] in range [0, 255]
        img = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return img

    def _generate_union_mask(self, image_tensor, top_k=10):
        """
        Generate union of top-K largest SAM masks for an image.
        
        Args:
            image_tensor: Image tensor [B, C, H, W] in [0, 255]
            top_k: Number of largest masks to include in union
            
        Returns:
            Binary mask tensor [1, 1, H, W] on same device
        """
        img_np = self._tensor_to_numpy_image(image_tensor)
        h, w = img_np.shape[:2]
        
        mask_generator = self._get_mask_generator()
        
        with torch.no_grad():
            masks = mask_generator.generate(img_np)
        
        if len(masks) == 0:
            # No masks found, return empty mask
            return torch.zeros(1, 1, h, w, device=self.device, dtype=torch.float32)
        
        # Sort masks by area (largest first) and take top-K
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)[:top_k]
        
        # Create union mask
        union_mask = np.zeros((h, w), dtype=np.float32)
        for m in masks_sorted:
            union_mask = np.maximum(union_mask, m['segmentation'].astype(np.float32))
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(union_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        return mask_tensor

    def compute_iou_score(self, source, generated, top_k=10):
        """
        Compute IoU (Intersection over Union) between SAM mask unions.
        
        Args:
            source: Source image tensor [B, C, H, W] in [0, 255]
            generated: Generated image tensor [B, C, H, W] in [0, 255]
            top_k: Number of largest masks to include in union
            
        Returns:
            IoU score (higher is better, in [0, 1])
        """
        with torch.no_grad():
            mask_source = self._generate_union_mask(source, top_k)
            mask_generated = self._generate_union_mask(generated, top_k)
            
            intersection = (mask_source * mask_generated).sum()
            union = ((mask_source + mask_generated) > 0).float().sum()
            
            if union == 0:
                return torch.tensor(1.0, device=self.device)  # Both empty = perfect match
            
            iou = intersection / union
        return iou

    def compute_dice_score(self, source, generated, top_k=10):
        """
        Compute Dice coefficient between SAM mask unions.
        
        Args:
            source: Source image tensor [B, C, H, W] in [0, 255]
            generated: Generated image tensor [B, C, H, W] in [0, 255]
            top_k: Number of largest masks to include in union
            
        Returns:
            Dice score (higher is better, in [0, 1])
        """
        with torch.no_grad():
            mask_source = self._generate_union_mask(source, top_k)
            mask_generated = self._generate_union_mask(generated, top_k)
            
            intersection = (mask_source * mask_generated).sum()
            sum_masks = mask_source.sum() + mask_generated.sum()
            
            if sum_masks == 0:
                return torch.tensor(1.0, device=self.device)  # Both empty = perfect match
            
            dice = (2.0 * intersection) / sum_masks
        return dice

    def mask_consistency_score(self, source, generated, mode='iou', top_k=10):
        """
        Compute mask-based consistency score.
        
        Args:
            source: Source image tensor
            generated: Generated image tensor
            mode: 'iou' for IoU score, 'dice' for Dice score
            top_k: Number of largest masks to include
            
        Returns:
            Consistency score (higher is better)
        """
        if mode == 'iou':
            return self.compute_iou_score(source, generated, top_k)
        elif mode == 'dice':
            return self.compute_dice_score(source, generated, top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'iou' or 'dice'.")
        
