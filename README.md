# NEB-YOLO

Lightweight fine-grained bird recognition based on improved YOLOv8.

## Overview

NEB-YOLO integrates three core techniques:
- **EMA Attention**: Efficient Multi-scale Attention in the Neck network for enhanced feature extraction
- **Group_Slim Pruning**: Structured channel pruning reducing parameters by 64%
- **Joint Distillation**: Combined BCKD (logic-based) and CWD (feature-based) knowledge distillation

## Performance

| Model | Parameters | mAP | Reduction |
|-------|-----------|-----|-----------|
| YOLOv8n baseline | 2,786,268 | 79.95% | - |
| NEB-YOLO | ~1,000,000 | 80.65% | 64% |

Dataset: CUB-200-2011

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python >= 3.8, PyTorch >= 1.8.0

## Project Structure

```
NEB-YOLO/
├── ultralytics/          # Core framework (v8.3.9)
│   ├── cfg/models/v8/    # Model configs
│   └── nn/modules/       # EMA module
├── train.py              # Training
├── compress.py           # Pruning
├── distill.py            # Distillation
├── detect.py             # Inference
├── val.py                # Validation
└── dataset/              # Data directory
```

## Usage

### 1. Train Teacher Network

```python
from ultralytics import YOLO

model = YOLO('yolov8-EMA.yaml').load('yolov8n.pt')
model.train(data='cub.yaml', epochs=300, batch=32, imgsz=640)
```

Or run:
```bash
python train.py
```

### 2. Prune Model

```bash
python compress.py
```

### 3. Distill Knowledge

```bash
python distill.py
```

### 4. Inference

```python
from ultralytics import YOLO

model = YOLO('runs/distill/exp/weights/best.pt')
results = model('bird.jpg')
```

## Model Configs

- Teacher: [yolov8-EMA.yaml](yolo-distill-deploy/yolo-distill-deploy/yolov8-EMA.yaml)
- Student: Standard YOLOv8n with Group_Slim pruning

## Citation

```bibtex
@article{neb-yolo2024,
  title={NEB-YOLO: Lightweight Fine-Grained Bird Recognition},
  author={},
  year={2024}
}
```

## License

AGPL-3.0
