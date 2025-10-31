# Duplicate Question Pair Detection

## Project Overview
This project aims to identify whether two given questions convey the same meaning.  
It’s a Natural Language Processing (NLP) and Machine Learning task inspired by the **Quora Question Pairs** challenge.  
The goal is to build a model that detects duplicate questions, helping to reduce redundant content on Q&A platforms like Quora and Stack Overflow.

---

## Objective
- Detect semantic similarity between two text-based questions.  
- Classify question pairs as *duplicate* or *non-duplicate*.  
- Enhance user experience by merging or filtering similar queries.  

---

## Dataset
The dataset used is the **Quora Question Pairs** dataset from [Kaggle](https://www.kaggle.com/competitions/quora-question-pairs).  
It contains over **400,000 question pairs** with labels:
- `1` → Duplicate  
- `0` → Not Duplicate  

---

## Methodology

### 1. Data Preprocessing
- Removal of punctuation, numbers, and special characters  
- Conversion to lowercase  
- Tokenization of questions  
- Stopword removal using NLTK  
- Lemmatization/Stemming for normalization  
- Handling missing or null values  

### 2. Feature Engineering
- **Basic NLP Features:** Word count, character count, common words count, etc.  
- **Fuzzy Matching Scores:** Similarity ratio, partial ratio, token set ratio, etc.  
- **Bag of Words (BoW) and TF-IDF Representations**  
- **Advanced Features:** Cosine similarity, Jaccard similarity, etc.  

### 3. Model Building
- Machine Learning algorithms such as:
  - Logistic Regression  
  - Random Forest Classifier  
  - XGBoost  
- Performance metrics used: Accuracy, Precision, Recall, F1-score  

### 4. Model Evaluation
- Dataset split into **training** and **testing** sets (e.g., 80–20 split).  
- Confusion Matrix and ROC-AUC score used for model validation.  

---

## Tools and Libraries Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK, FuzzyWuzzy  
- Matplotlib, Seaborn  
- XGBoost  

---

## Results
- The model achieved high accuracy in detecting duplicate question pairs.  
- Incorporating text similarity features and BoW/TF-IDF improved classification performance.  
- The final model was saved as `.pkl` using `joblib` for future deployment.  



   git clone https://github.com/yourusername/Duplicate-Question-Pair-Detection.git

