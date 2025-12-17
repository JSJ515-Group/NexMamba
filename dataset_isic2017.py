"""
ISIC2017 dataset + high-quality preprocessing and augmentation
- RGB input preserved
- Uses albumentations for safe, synchronized augmentations
- Keeps mask integrity (nearest interpolation, binary remap 0/255 -> 0/1)
- Resize with aspect-ratio preserving LongestMaxSize + PadIfNeeded (reflect padding)
- Normalization with ImageNet mean/std (recommended for pretrained encoders)
- Returns dict: { 'image': Tensor[C,H,W], 'label': Tensor[H,W], 'case_name': str }

Usage:
    train_set = ISIC2017Dataset(root='dataset/isic2017', split='train', img_size=224)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=ISIC2017Dataset.collate_fn)

Dependencies: albumentations, cv2, numpy, torch
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    raise ImportError("This module requires albumentations and albumentations.pytorch. Install: pip install albumentations")


# -------------------------
# Helpers
# -------------------------

def _find_mask_path(mask_dir: str, case_name: str) -> Optional[str]:
    """Try common mask filename variants."""
    candidates = [
        f"{case_name}_segmentation.png",
        f"{case_name}_Segmentation.png",
        f"{case_name}.png",
        f"{case_name}_mask.png",
        f"{case_name}_mask.PNG",
    ]
    for c in candidates:
        p = os.path.join(mask_dir, c)
        if os.path.exists(p):
            return p
    return None


# -------------------------
# Augmentations / transforms
# -------------------------

def get_transforms(img_size: int, split: str = "train") -> A.Compose:
    """Return an albumentations Compose for train/val.
    - LongestMaxSize then PadIfNeeded to keep aspect and avoid black pad (use reflect)
    - Light geometric + photometric augmentations for train
    - No blur that can destroy edges
    """
    base = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT)
    ]

    if split == "train":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT, p=0.6),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            # small gaussian noise (don't alter edges much)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            # Cutout/mask-preserving coarse dropout sometimes helps
            A.CoarseDropout(max_holes=1, max_height=int(img_size*0.1), max_width=int(img_size*0.1), min_holes=0, fill_value=0, p=0.2),
        ]
    else:
        aug = []

    # ToTensorV2 converts image->C,H,W float32 (0-1) and mask to long or uint8
    post = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]

    return A.Compose(base + aug + post)



# def get_transforms(img_size: int, split: str = "train") -> A.Compose:
#     base = [
#         A.LongestMaxSize(max_size=img_size),
#         A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_REFLECT),
#     ]
#
#     if split == "train":
#         aug = [
#             A.HorizontalFlip(p=0.5),
#
#             A.ShiftScaleRotate(
#                 shift_limit=0.04,
#                 scale_limit=0.12,     # 强 scale
#                 rotate_limit=25,
#                 interpolation=cv2.INTER_LINEAR,
#                 border_mode=cv2.BORDER_REFLECT,
#                 p=0.7
#             ),
#
#             # ★ SOTA 常用
#             A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
#             A.ElasticTransform(alpha=20, sigma=6, alpha_affine=6, p=0.2),
#
#             A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
#             A.HueSaturationValue(8, 15, 10, p=0.3),
#
#             A.GaussNoise(var_limit=(5, 30), p=0.15),
#         ]
#     else:
#         aug = []
#
#     post = [
#         A.Normalize(mean=(0.485, 0.456, 0.406),
#                     std=(0.229, 0.224, 0.225)),
#         ToTensorV2()
#     ]
#
#     return A.Compose(base + aug + post)

# -------------------------
# Dataset
# -------------------------
class ISICDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 img_size: int = 256,
                 file_list: Optional[str] = None,
                 return_case_name: bool = True) -> None:
        """
        root: path to dataset root, expects subfolders: <root>/images and <root>/masks
        split: used only for choosing augmentations; dataset structure is read from folders
        file_list: optional path to txt file with one case name per line (without extension). If not provided, will list images folder.
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.return_case_name = return_case_name

        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        if file_list is not None:
            with open(file_list, 'r') as f:
                self.cases = [line.strip() for line in f.readlines()]
        else:
            # list files in image_dir, keep filename without ext
            imgs = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.cases = sorted([os.path.splitext(f)[0] for f in imgs])

        if len(self.cases) == 0:
            raise RuntimeError(f"No image files found in {self.image_dir}")

        self.transform = get_transforms(img_size, split)

    def __len__(self) -> int:
        return len(self.cases)

    def _load_image(self, case: str) -> np.ndarray:
        # accept .jpg/.png
        for ext in ('.jpg', '.png', '.jpeg'):
            p = os.path.join(self.image_dir, case + ext)
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {p}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                return img
        raise FileNotFoundError(f"Image not found for case {case} in {self.image_dir}")

    def _load_mask(self, case: str) -> np.ndarray:
        mask_path = _find_mask_path(self.mask_dir, case)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for case {case} in {self.mask_dir}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        # remap 255 -> 1, keep as uint8
        mask = (mask > 127).astype(np.uint8)
        return mask

    def __getitem__(self, idx: int):
        case = self.cases[idx]
        img = self._load_image(case)
        mask = self._load_mask(case)

        # albumentations expects image uint8 for some ops, but ToTensorV2 will handle float32 too.
        # We pass float in range [0,1] and albumentations will work; to be safe, convert to 0-255 for certain ops
        # Compose will handle both, but ensure types are correct

        augmented = self.transform(image=(img * 255).astype(np.uint8), mask=(mask * 255).astype(np.uint8))
        image = augmented['image']  # Tensor C,H,W float normalized
        mask = augmented['mask']    # Tensor H,W (uint8)

        # mask may be 0/255 after augmentation; convert to 0/1 long
        if isinstance(mask, torch.Tensor):
            mask = (mask > 127).long()
            # drop channel if exists
            if mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.squeeze(0)
        else:
            mask = torch.from_numpy((mask > 127).astype(np.int64))

        out = {
            'image': image,        # torch.FloatTensor C,H,W (normalized)
            'label': mask,         # torch.LongTensor H,W
            'case_name': case
        }
        return out

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        images = torch.stack([b['image'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        case_names = [b['case_name'] for b in batch]
        return {'image': images, 'label': labels, 'case_name': case_names}


# -------------------------
# Loader factory
# -------------------------
def get_loaders(root: str, img_size: int = 256, batch_size: int = 8, num_workers: int = 4):
    train_ds = ISICDataset(root=os.path.join(root, ''), split='train', img_size=img_size)
    val_ds = ISICDataset(root=os.path.join(root, ''), split='val', img_size=img_size)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ISICDataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=ISICDataset.collate_fn)

    return train_loader, val_loader

