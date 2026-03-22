# Power-Quality-Disturbances

Deep transfer learning for automated power quality disturbances detections in solar farms



\# ⚡ Power Quality Disturbance Classification using Deep Learning



This repository presents a deep learning-based framework for detecting and classifying \*\*Power Quality Disturbances (PQDs)\*\* using time-series signal transformation and image-based models such as \*\*ResNet, VGG, and CNN\*\*.



\---



\## 📌 Project Overview



Power Quality Disturbances (PQDs) are deviations in voltage/current signals that affect power system performance. This project focuses on:



\* Extracting disturbance patterns from time-series data

\* Converting signals into images using \*\*Gramian Angular Field (GAF)\*\*

\* Applying deep learning models for classification

\* Evaluating performance using ROC curves and classification metrics



\---



\## 🧠 Methodology



\### 1. Data Processing



\* Input data from CSV files (train/test datasets)

\* Feature extraction using sliding window technique

\* Normalization using MinMaxScaler



\### 2. Signal to Image Conversion



\* Time-series signals transformed into images using:



&#x20; \* Gramian Angular Field (GAF)



\### 3. Deep Learning Models



The following models are implemented:



\* ✅ ResNet50

\* ✅ VGG (for ROC comparison)

\* ✅ Custom CNN



\### 4. Training Strategies



Multiple train-test splits are used:



\* 60% – 40%

\* 70% – 30%

\* 80% – 20%



With:



\* Additional validation split (e.g., 80/20 from training set)

\* Stratified sampling for class balance



\### 5. Evaluation Metrics



\* Accuracy, Loss

\* Confusion Matrix

\* Classification Report

\* ROC Curve and AUC



\---



\## 📂 Repository Structure



```

..

├── Data/                                         # Dataset files

│   ├── Train\_data.csv

│   └── Test\_data.csv

│

├── CNN/                                          # CNN experiments

│   ├── CNN\_60\_40\_with\_validation\_Type1.py

│   └── CNN\_60\_40\_with\_validation\_Type4.py   ...

│

├── ResNet/                                       # ResNet experiments

│   ├── ResNet\_60\_40\_with\_validation\_Type1.py

│   └── ResNet\_60\_40\_with\_validation\_Type4.py  ...

│

├── VGG/                                          # VGG experiments

│   ├── VGG\_60\_40\_with\_validation\_Type1.py

│   └── VGG\_60\_40\_with\_validation\_Type4.py   ...

│

├── ROC/                                          # ROC analysis with K

│   ├── ResNet.py

│   └── VGG.py

│

├── requirements.txt                              # Python dependencies

└── README.md                                     # Project overview and instructions

\---



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/NirmalShiroya/Power-Quality-Disturbances.git

cd Power-Quality-Disturbances

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## 🚀 Usage



\### Run ResNet Model (Example)



```bash

python ResNet.py

```



\### Run specific experiment (example)



```bash

python ResNet\_60\_40\_with\_validation\_Type\_1.py

```



\---



\## 📊 Key Features



\* ✔ Time-series to image transformation (GAF)

\* ✔ Deep learning-based classification

\* ✔ Multiple disturbance types handling

\* ✔ ROC curve analysis using K-Fold cross-validation

\* ✔ Class imbalance handling using class weights



\---



\## 📈 Results



The project generates:



\* Training \& Validation Accuracy/Loss plots

\* Confusion Matrix visualization

\* ROC Curve with AUC comparison

\* Pattern matching visualization (test vs closest training pattern)



\---



\## 🔬 Example Disturbance Types



\* Voltage Sag

\* Voltage Swell

\* Harmonics

\* Waveform Distortion



(Binary classification is applied per disturbance type)



\---



\## 📚 Citation



If you use this work, please cite:



```bibtex

@misc{power\_quality\_disturbance\_dl,

&#x20; author = {Nirmal Shiroya},

&#x20; title = {Power Quality Disturbance Classification using Deep Learning},

&#x20; year = {2025},

&#x20; url = {https://github.com/NirmalShiroya/Power-Quality-Disturbances}

}

```



\---



\## ⚠️ Notes



\* Update dataset paths in scripts (currently local paths are used)

\* Ensure CSV files are placed inside the `Data/` folder

\* GPU is recommended for faster training (TensorFlow)



\---



\## 👤 Author



\*\*Nirmal Shiroya\*\*

GitHub: https://github.com/NirmalShiroya



\---



\## ⭐ Acknowledgements



\* Signal processing and power systems research community

\* Libraries: TensorFlow, Scikit-learn, PyTS



