# 🤖 Machine Learning Classifier Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Classifiers-orange?style=for-the-badge&logo=scikitlearn)

This project compares **multiple machine learning classifiers** on a dataset to identify the **best performing model** based on:  
✅ Accuracy  
✅ F1 Score  
✅ AUC Score  

It also applies **resampling techniques** to handle class imbalance and **standard scaling** for feature normalization.  

---

## 📂 Project Structure
- `ml.py` – Main script that processes the dataset and evaluates classifiers.  
- `requirements.txt` – List of dependencies.  

---

## ⚙️ Setup

### 🔑 Prerequisites
- Python **3.x**  

### 📦 Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1️⃣ Prepare Dataset
Place your CSV file (e.g., `log2.csv`) in the **project directory**.  

### 2️⃣ Run the Script
```bash
python ml.py
```

### 3️⃣ Output
- 📊 Processed results and analysis printed in the console.  
- 📈 Plots and logs generated in the specified directories.  

---

## 📦 Dependencies
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `imbalanced-learn`  
- `seaborn`  
- `matplotlib`  

---

## 📁 Directory Structure
Expected structure:
```
project-directory/
├── README.md
├── requirements.txt
├── ml.py
├── log2.csv
```

---

## 🤝 Contributing
Contributions are welcome!  
Fork this repository, make improvements, and submit a pull request.  

---

## 📜 License
This project is licensed under the **MIT License**.  
