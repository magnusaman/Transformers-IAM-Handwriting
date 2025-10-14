# Handwritten Paragraph-to-Text Transformer

A custom CNN Encoder + Transformer Decoder model for converting handwritten paragraph images into text using the IAM Forms Dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates hands-on understanding of Transformer architectures for OCR-style tasks. It trains a deep learning model to recognize handwritten text from paragraph images using:

- **Custom CNN Encoder** for visual feature extraction
- **Transformer Decoder** for sequence generation
- **Teacher Forcing** training strategy
- **Character-level tokenization**

### Key Features

- ✅ Proper IAM Forms dataset preprocessing with corrected cropping
- ✅ ~50M parameter model with attention mechanisms
- ✅ Character Error Rate (CER) and Word Error Rate (WER) evaluation
- ✅ Greedy decoding for inference
- ✅ Complete training pipeline with checkpointing
- ✅ GPU-optimized for RunPod/Colab environments

## 📁 Dataset

**IAM Handwritten Forms Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset)
- ~4GB of handwritten form images
- Each image contains:
  - Top 20%: Printed text (ground truth labels)
  - Middle 20-65%: Handwritten text (model input)
  - Bottom 35%: Signature field

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision transformers easyocr jiwer matplotlib tqdm kaggle
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/magnusaman/Transformers-IAM-Handwriting.git
cd Transformers-IAM-Handwriting
```

2. **Add Kaggle credentials**
```bash
# Place your kaggle.json in the workspace directory
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. **Run the notebook**
```bash
jupyter notebook transformers.ipynb
```

### Training

Simply run all cells in `transformers.ipynb` sequentially. The notebook handles:

1. Dataset download and extraction
2. Image preprocessing with EasyOCR
3. Vocabulary building
4. Model training (15 epochs)
5. Evaluation with CER/WER metrics
6. Inference on test images

**Expected Training Time:** ~2-3 hours on GPU (RunPod/Colab)

## 🏗️ Architecture

### CNN Encoder
```
Input: (B, 1, 128, 512)
  ↓
Conv2d(1, 64) + ReLU + MaxPool
  ↓
Conv2d(64, 128) + ReLU + MaxPool
  ↓
Conv2d(128, 256) + ReLU + MaxPool
  ↓
Conv2d(256, 512) + ReLU + MaxPool
  ↓
Linear Projection
  ↓
Output: (B, 64, 512)  # Visual features
```

### Transformer Decoder
```
Visual Features + Text Embeddings
  ↓
Positional Encoding
  ↓
6x TransformerDecoderLayer
  - 8 attention heads
  - 2048 dim feedforward
  - Causal masking
  ↓
Linear(512, vocab_size)
  ↓
Output: Character predictions
```

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~50M |
| d_model | 512 |
| Attention Heads | 8 |
| Decoder Layers | 6 |
| Vocabulary Size | 200 characters |
| Max Sequence Length | 256 tokens |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | OneCycleLR (max_lr=5e-4) |
| Batch Size | 16 |

## 📈 Training Details

- **Loss Function:** CrossEntropyLoss (ignoring PAD tokens)
- **Training Strategy:** Teacher forcing with causal masking
- **Gradient Clipping:** max_norm=1.0
- **Train/Val Split:** 90/10
- **Epochs:** 15
- **Learning Rate:** 1e-4 → 5e-4 → 1e-6 (OneCycleLR)

## 🎯 Evaluation

The model is evaluated using:

- **CER (Character Error Rate):** Character-level accuracy
- **WER (Word Error Rate):** Word-level accuracy

Metrics are calculated using the `jiwer` library on 100 validation samples.

## 📦 Output Artifacts

After training, the following files are saved:

```
/workspace/
├── best_model.pth          # Best model checkpoint
├── final_model.pth         # Final model + config
├── tokenizer.pkl           # Character tokenizer
├── training_curve.png      # Loss visualization
└── processed/
    ├── images/             # Preprocessed handwriting images
    └── labels.txt          # Ground truth labels
```

## 🔮 Inference

```python
from PIL import Image
import torch

# Load model and tokenizer
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict from image
predicted_text = predict_from_image_path(
    'path/to/handwritten.png',
    model,
    tokenizer,
    device='cuda'
)
print(predicted_text)
```

## 📝 Project Structure

```
.
├── transformers.ipynb      # Main notebook (all-in-one)
├── README.md               # This file
├── .gitignore             # Git ignore rules
└── kaggle.json            # Kaggle API credentials (not tracked)
```

## 🛠️ Technical Highlights

### Fixed Cropping Issue
The original 50-50 image split was cutting through handwritten text. This implementation uses:
- Top 20% for printed labels (OCR'd with EasyOCR)
- Middle 20-65% for handwritten input
- Bottom 35% discarded (signature/whitespace)

### Tokenization Strategy
- Character-level vocabulary (200 most common chars)
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Maximum sequence length: 256 characters

### Memory Optimization
- Images resized to 128x512 (grayscale)
- Batch size 16 optimized for 16GB GPU
- Gradient accumulation ready (if needed)

## 🚧 Future Improvements

- [ ] Implement beam search for better predictions
- [ ] Add data augmentation (rotation, noise, blur)
- [ ] Experiment with TrOCR architecture
- [ ] Add attention visualization
- [ ] Fine-tune on specific writing styles
- [ ] Deploy as web API with FastAPI

## 🎓 Educational Context

This project was built to demonstrate understanding of:
- Transformer architectures for vision-language tasks
- CNN feature extraction for images
- Autoregressive sequence generation
- Teacher forcing training
- Character-level language modeling
- OCR evaluation metrics

Perfect for academic projects, portfolio demonstrations, or learning Transformers!

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [TrOCR: Transformer-based OCR](https://arxiv.org/abs/2109.10282) - OCR with Transformers
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your training results

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- IAM Handwriting Database for the dataset
- Kaggle for hosting the dataset
- RunPod for GPU infrastructure
- PyTorch and Hugging Face communities

---

**Built with ❤️ for deep learning education**

*For questions or issues, please open an issue on GitHub*
