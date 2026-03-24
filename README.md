# ⚡ Power Quality Disturbance Detection using Deep Learning

**Deep Transfer Learning Framework for Automated Power Quality Disturbance Classification in Solar Farms**

---

## 🔍 Overview

Power Quality Disturbances (PQDs) are deviations in voltage or current waveforms that can significantly impact the stability and efficiency of power systems—especially in renewable energy environments like solar farms.

This project presents a **deep learning-based classification framework** that transforms time-series electrical signals into image representations and leverages **transfer learning models** to accurately detect and classify disturbance types.

---

## ✨ Key Features

* ✅ Time-series to image transformation using **Gramian Angular Field (GAF)**
* ✅ Deep learning models: **ResNet50, VGG, Custom CNN**
* ✅ Transfer learning for improved classification performance
* ✅ ROC-AUC evaluation with K-Fold cross-validation
* ✅ Class imbalance handling using class weights
* ✅ Robust evaluation across multiple train-test splits
* ✅ Pattern similarity visualization (test vs training signals)

---

## 🧠 Methodology

### 🔹 Data Processing

* Input: Time-series data from CSV files
* Sliding window technique for feature extraction
* Data normalization using **MinMaxScaler**

### 🔹 Signal-to-Image Transformation

* Time-series signals converted into images using:

  * **Gramian Angular Field (GAF)**
* Enables the use of image-based deep learning models

### 🔹 Deep Learning Models

* **ResNet50** (primary model with transfer learning)
* **VGG** (used for comparative ROC analysis)
* **Custom CNN** (baseline model)

### 🔹 Training Strategy

* Train-test splits:

  * 60% – 40%
  * 70% – 30%
  * 80% – 20%
* Additional validation split (e.g., 80/20 within training set)
* Stratified sampling for balanced class distribution

### 🔹 Evaluation Metrics

* Accuracy & Loss curves
* Confusion Matrix
* Classification Report
* ROC Curve & AUC Score

---

## 📁 Repository Structure

```id="z3kqpl"
Power-Quality-Disturbances/
│
├── Metadata/                      # Dataset files
│   ├── Train_data.csv
│   └── Test_data.csv
│
├── CNN/                           # CNN model experiments
├── ResNet/                        # ResNet model experiments
├── VGG/                           # VGG model experiments
│
├── ROC/                           # ROC & K-Fold analysis
│   ├── ResNet.py
│   └── VGG.py
│
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quick Start

### 🔧 Installation

```bash id="k2ld9a"
# Clone the repository
git clone https://github.com/NirmalShiroya/Power-Quality-Disturbances.git
cd Power-Quality-Disturbances

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run ResNet Model

```bash id="x91mqp"
python ROC/ResNet.py
```

### Run Specific Experiment

```bash id="v7af0c"
python ResNet/ResNet_60_40_with_validation_Type1.py
```

---

## 📊 Results & Outputs

The project generates:

* 📈 Training & validation accuracy/loss plots
* 📊 Confusion matrix visualization
* 📉 ROC curves with AUC comparison
* 🔍 Pattern similarity visualization between signals

---

## 🔬 Disturbance Types Covered

* Voltage Sag [Disturbance Type 1 ]
* Voltage Swell [Disturbance Type 1 ]
* Waveform Distortion [ Disturbance Type 4 ]

> Binary classification is performed for each disturbance type.

---

## 💡 Key Contributions

* Developed a **hybrid signal processing + deep learning pipeline**
* Applied **GAF transformation** to leverage CNN-based architectures
* Demonstrated **transfer learning effectiveness** in power systems
* Built a **robust evaluation framework** using ROC & cross-validation

---

## 📚 Citation

If you use this work, please cite:

```bibtex id="9j2nax"
@misc{shiroya2025pqd,
  author = {Nirmal Shiroya},
  title = {Power Quality Disturbance Detection using Deep Learning},
  year = {2025},
  url = {https://github.com/NirmalShiroya/Power-Quality-Disturbances}
}
```

---

## ⚠️ Notes

* Update dataset paths in scripts if needed
* Ensure CSV files are placed inside the `Metadata/` folder
* GPU (CUDA-enabled) is recommended for faster training

---

## 👤 Author

**Nirmal Shiroya**
🔗 GitHub: https://github.com/NirmalShiroya
🔗 LinkedIn: https://linkedin.com/in/nirmalshiroya

---

## ⭐ Acknowledgements

* GENESIS Research Lab & Neuville Data Grid
* Power systems & signal processing research community
* Open-source libraries:

  * TensorFlow
  * Scikit-learn
  * PyTS

---

## 🚀 Future Improvements

* Multi-class classification for simultaneous disturbance detection
* Real-time deployment on smart grid systems
* Integration with IoT-based monitoring systems
* Advanced architectures (Transformer-based models)

---

