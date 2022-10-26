import torch
import numpy as np


def crop_pad(img, size, center=None):
    if torch.is_tensor(img):
        H, W = img.shape[-2:]
    else:
        H, W = img.shape[:2]
    h, w = size
    if center is None:
        y, x = H // 2, W // 2
    else:
        y, x = center
    half_w = w // 2
    half_h = h // 2

    y0 = int(max(0, half_h - y))
    y1 = int(max(0, y - half_h))
    h = int(min(h, H - y1) - y0)

    x0 = int(max(0, half_w - w))
    x1 = int(max(0, x - half_w))
    w = int(min(w, W - x1) - x0)

    if torch.is_tensor(img):
        r = torch.zeros(tuple(img.shape[:-2]) + size, dtype=img.dtype, device=img.device)
        r[..., y0:y0 + h, x0:x0 + w] = img[..., y1:y1 + h, x1:x1 + w]
    else:
        r = np.zeros_like(img, shape=size + img.shape[2:])
        r[y0:y0 + h, x0:x0 + w] = img[y1:y1 + h, x1:x1 + w]

    return r
