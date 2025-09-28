import torch, cv2, numpy as np, mmcv, mmdet, mmrotate
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("cv2:", cv2.__version__, "| numpy:", np.__version__)
print("mmcv:", mmcv.__version__, "| mmdet:", mmdet.__version__, "| mmrotate:", getattr(mmrotate, "__version__", "git-1.x"))