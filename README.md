# breast-cancer-tabnet-classifier

This project trains a TabNet model on the Breast Cancer Wisconsin dataset to predict malignancy.

## Project Structure 

breast-cancer-tabnet-classifier/
│
├── data/ # Place dataset(s) here
│ └── breast-cancer-wisconsin-data_data.csv
├── notebooks/
│ └── exploratory_analysis.ipynb # Data exploration notebook
├── src/
│ └── breast_cancer_tabnet.py # Training + evaluation script
├── README.md
├── requirements.txt # Dependencies


# Dataset
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign)
- **Features**: 30 numerical features describing cell nuclei characteristics.

### How to Add Data
1. Download the CSV dataset from Kaggle or UCI ML Repository.
2. Place the file in the `data/` folder:

# Getting Started 
### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-tabnet-classifier.git
cd breast-cancer-tabnet-classifier
```

### 2. Install dependencies 
```bash
pip install -r requirements.txt
```

### 3. Run Exploratory Analysis 
```bash
python notebooks/exploratory_analysis.py
```

### 4. Train & Evaluate the Model
```bash
python src/breast_cancer_tabnet.py
```
