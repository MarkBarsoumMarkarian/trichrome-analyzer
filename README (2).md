# 🔬 Trichrome Fibrosis Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

**AI-powered quantification of renal interstitial fibrosis from Masson's Trichrome-stained kidney biopsies**

This tool combines traditional computer vision with machine learning to automatically segment and quantify fibrosis in kidney biopsy images, providing pathologists with objective, reproducible measurements.

![Demo](docs/demo.gif)

---

## ✨ Features

### Core Capabilities
- 🎯 **Automated Segmentation** - Detects blue/green collagen fibrosis using color-space analysis
- 🔵 **Glomeruli Exclusion** - Automatically identifies and excludes glomeruli (both healthy and sclerosed)
- 🤖 **ML Correction** - Machine learning models refine raw segmentation for improved accuracy
- 🎨 **Color Normalization** - Vahadane stain normalization handles variations in staining intensity
- 📊 **Clinical Grading** - Automatically categorizes fibrosis as Minimal/Mild/Moderate/Severe

### Advanced Features
- 🔍 **Explainability** - Feature importance and severity heatmaps
- ⚡ **Batch Processing** - Process entire folders efficiently
- 📈 **Clinical Validation** - Built-in metrics (MAE, R², Cohen's Kappa)
- 🌐 **Web Interface** - User-friendly Streamlit app
- 📥 **Export Reports** - Download results as text reports or segmentation masks

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trichrome-analyzer.git
cd trichrome-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Web Interface

```bash
streamlit run trichrome_app.py
```

Open your browser to `http://localhost:8501` and upload a trichrome-stained image!

---

## 📖 Usage

### 1. Web Interface (Recommended for Single Images)

The easiest way to analyze images:

```bash
streamlit run trichrome_app.py
```

1. Upload a Masson's Trichrome-stained kidney biopsy image
2. View real-time segmentation and analysis
3. Download reports and masks

### 2. Command Line Interface

#### Train a Model

Prepare training data with filenames containing the ground truth percentage:
```
training_data/
├── tri_15.5%.jpg    # 15.5% fibrosis
├── tri_32.1%.jpg    # 32.1% fibrosis
└── tri_8.3%.jpg     # 8.3% fibrosis
```

Train the model:
```bash
python trichrome_core.py train --data training_data --model my_model.pkl
```

#### Predict on Single Image

```bash
python trichrome_core.py predict --data sample.jpg --model trichrome_model.pkl
```

#### Batch Process Folder

```bash
python trichrome_core.py batch --data test_images/ --output results.csv
```

#### Clinical Validation

```bash
python trichrome_core.py validate --data validation.csv --model trichrome_model.pkl
```

### 3. Python API

```python
from trichrome_core import TrichromeAnalyzer
import cv2

# Initialize analyzer
analyzer = TrichromeAnalyzer()

# Load pre-trained model (optional)
analyzer.load_model('trichrome_model.pkl')

# Analyze image
image = cv2.imread('kidney_biopsy.jpg')
raw_percent, ml_prediction, mask = analyzer.predict(image)

print(f"Raw segmentation: {raw_percent:.2f}%")
print(f"ML-corrected: {ml_prediction:.2f}%")

# Generate heatmap
heatmap = analyzer.generate_heatmap(image, mask)

# Get feature explanations
explanation = analyzer.explain_features(image)
```

---

## 🗂️ Project Structure

```
trichrome-analyzer/
├── trichrome_app.py           # Streamlit web interface
├── trichrome_core.py          # Core analysis engine
├── trichrome_advanced.py      # Advanced features (U-Net, ensemble, active learning)
├── data_collection.py         # Helper script to collect training images
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── quality_assurance/         # QA and diagnostic tools
│   ├── diagnostic_script.py   # Analyze training data quality
│   └── segmentation_visualizer.py  # Visualize segmentation results
│
├── models/                    # Saved models (not in repo)
│   └── trichrome_model.pkl
│
├── examples/                  # Example images and notebooks
│   ├── sample_images/
│   └── tutorial.ipynb
│
└── docs/                      # Documentation
    ├── api_reference.md
    ├── clinical_validation.md
    └── training_guide.md
```

---

## 🎨 How It Works

### 1. Color-Based Segmentation

Masson's Trichrome staining produces distinct colors:
- 🔴 **Red/Pink** - Normal tissue (muscle, cytoplasm)
- 🔵 **Blue/Green** - Collagen fibrosis (what we measure)
- ⚪ **White** - Background and some glomeruli

The analyzer detects blue/green pixels in HSV color space while excluding glomeruli.

### 2. Glomeruli Exclusion

Uses Hough Circle Transform to detect:
- Red/pink circular structures (healthy glomeruli)
- Blue/green circular structures (sclerosed glomeruli)

Both are excluded from fibrosis measurements.

### 3. Machine Learning Refinement

Extracts 14 features from each image:
- Color statistics (HSV, LAB)
- Spatial distribution
- Region properties

Trains regression models (Ridge, Random Forest, Gradient Boosting) to correct systematic biases.

### 4. Vahadane Normalization (Optional)

Normalizes staining variations across different labs/scanners using sparse NMF decomposition.

---

## 📊 Clinical Grading

| Grade | Fibrosis % | Interpretation |
|-------|-----------|----------------|
| 🟢 **Minimal** | < 10% | Minimal interstitial scarring |
| 🟡 **Mild** | 10-25% | Mild interstitial scarring |
| 🟠 **Moderate** | 25-50% | Moderate interstitial damage |
| 🔴 **Severe** | > 50% | Extensive interstitial scarring |

---

## 🔬 Training Your Own Model

### Step 1: Collect Training Data

Use the `data_collection.py` script to gather labeled images:

```bash
python data_collection.py
```

Modify the script to point to your image directories. Images must:
- Start with "tri" or "Tri"
- Contain "%" in filename
- Have the fibrosis percentage in the filename (e.g., `tri_15.5%.jpg`)

### Step 2: Quality Check Your Data

Before training, diagnose your dataset:

```bash
# Check label distribution and segmentation quality
python quality_assurance/diagnostic_script.py training_data/

# Visualize worst segmentation cases
python quality_assurance/segmentation_visualizer.py training_data/
```

This helps identify:
- Systematic biases
- Inconsistent labels
- Segmentation failures

### Step 3: Train the Model

```bash
python trichrome_core.py train --data training_data/ --model my_model.pkl
```

Optional flags:
- `--no-normalize` - Disable Vahadane color normalization
- `--model-type Ridge` - Force specific model type (Ridge, RandomForest, GradientBoosting)

### Step 4: Validate

Create a validation CSV with columns `filepath` and `ground_truth`:

```csv
filepath,ground_truth
test/tri_12.5%.jpg,12.5
test/tri_34.2%.jpg,34.2
```

Run validation:

```bash
python trichrome_core.py validate --data validation.csv --model my_model.pkl
```

---

## 🧪 Quality Assurance Tools

### Diagnostic Script

Analyzes training data quality:

```bash
python quality_assurance/diagnostic_script.py training_data/
```

**Outputs:**
- Label distribution (Minimal/Mild/Moderate/Severe)
- Mean absolute error across samples
- Systematic bias detection
- Identification of worst predictions

### Segmentation Visualizer

Visualizes what the algorithm detects:

```bash
# Visualize single image
python quality_assurance/segmentation_visualizer.py single image.jpg

# Test different HSV ranges
python quality_assurance/segmentation_visualizer.py test image.jpg

# Batch visualize worst cases
python quality_assurance/segmentation_visualizer.py training_data/
```

**Outputs:**
- Original image
- Blue/green detection
- Glomeruli masks
- Final fibrosis mask
- Color overlay

---

## 🚀 Advanced Features

### U-Net Deep Learning Segmentation

Train a U-Net model for pixel-wise segmentation:

```bash
python trichrome_advanced.py train-unet --images data/images --masks data/masks --epochs 50
```

### Ensemble Models

Combine multiple models for improved accuracy:

```bash
python trichrome_advanced.py ensemble \
  --traditional model.pkl \
  --unet unet.pth \
  --val validation.csv
```

### Active Learning

Intelligently select samples for labeling:

```bash
python trichrome_advanced.py active-learn \
  --ensemble ensemble.pkl \
  --unlabeled unlabeled_images/ \
  --output to_label.txt
```

---

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- OpenCV 4.5+
- NumPy
- scikit-learn
- SciPy
- Pillow

### Web Interface
- Streamlit 1.28+

### Advanced Features (Optional)
- PyTorch (for U-Net)
- pandas (for batch processing)
- matplotlib (for visualization)

See `requirements.txt` for complete list.

---

## 📈 Performance

Typical performance on validation datasets:

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 3-5% |
| R² Score | 0.85-0.92 |
| Cohen's Kappa | 0.75-0.85 |
| Processing Time | 1-3 seconds/image |

Results vary based on:
- Image quality
- Staining consistency
- Training data size and quality

---

## ⚠️ Clinical Disclaimer

**FOR RESEARCH USE ONLY**

This tool is provided for research purposes and is **not** intended for clinical diagnosis. Results should be:
- ✅ Verified by a qualified pathologist
- ✅ Used as a supplementary tool, not sole basis for decisions
- ✅ Validated on your specific dataset before use

The developers assume no liability for clinical decisions made using this software.

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- 🎯 Improved segmentation algorithms
- 🧠 Additional ML model architectures
- 🔬 Clinical validation studies
- 📚 Documentation improvements
- 🐛 Bug fixes

Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{trichrome_analyzer,
  title = {Trichrome Fibrosis Analyzer: AI-Powered Quantification of Renal Interstitial Fibrosis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/trichrome-analyzer}
}
```

---

## 🙏 Acknowledgments

- Vahadane normalization algorithm based on [Vahadane et al., 2016](https://ieeexplore.ieee.org/document/7460968)
- Inspired by digital pathology research at [Institution Name]
- Built with ❤️ for the nephropathology community

---

## 📞 Contact & Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/trichrome-analyzer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/trichrome-analyzer/discussions)
- 📧 **Email**: your.email@example.com

---

## 🗺️ Roadmap

- [ ] Support for additional staining types (PAS, H&E)
- [ ] Integration with whole-slide imaging (WSI) formats
- [ ] Cloud deployment options
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Real-time collaboration features

---

<div align="center">

**Made with 🔬 by researchers, for researchers**

[⬆ Back to Top](#-trichrome-fibrosis-analyzer)

</div>
