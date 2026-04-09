# 🧠 Sentiment Analysis for Mental Health Monitoring

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-success)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-89.13%25-brightgreen)

## 📌 Project Overview
This project is an **Advanced AI NLP Classification Pipeline** developed to accurately identify and monitor mental health statuses based on textual statements (e.g., social media texts, tweets). Early detection of psychological conditions via digital text is crucial for scaling psychiatric support and providing timely interventions.

**Target Classes (7 Categories):**
Normal | Depression | Suicidal | Anxiety | Bipolar | Stress | Personality disorder

## 🚀 Key Features & Methodology
1. **Smart Text Preprocessing:** Handled noise reduction (URLs, punctuation) while **strictly retaining negation words** (e.g., *not, don't*) to preserve the core sentiment context.
2. **Data Balancing:** Addressed severe class imbalance using mathematical oversampling (`sklearn.utils.resample`), ensuring exactly **3,269 samples per class**.
3. **Feature Engineering:** Utilized **TF-IDF Vectorization** with N-grams (`ngram_range=(1, 2)`) to capture Bigrams and contextual semantics.
4. **Model Configuration:** Deployed a highly tuned **Logistic Regression** classifier (`max_iter=2000`) optimized for high-dimensional, sparse text matrices.

## 📊 Final Performance Metrics
The model was evaluated on a massive unseen dataset (**22,881 samples**).

* **Overall Accuracy:** **89.13%**
* **Notable F1-Scores:**
  * Personality disorder: **0.98**
  * Bipolar: **0.96**
  * Anxiety: **0.94**
  * Stress: **0.92**

*(Check the attached Jupyter Notebook for the full Multi-class ROC Curve and Classification Report).*

## 📁 Repository Structure
* `Mental_Health_Sentiment_Analysis.ipynb`: The complete end-to-end Python codebase (EDA, Preprocessing, Training, Evaluation).
* `Project_Presentation.pdf`: The official presentation slides detailing the clinical context, architecture, and roadmap.
* `requirements.txt`: List of dependencies required to run the pipeline.

## 🔗 Dataset
The dataset used for this project is publicly available on Kaggle: 
[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

## 👥 Team Members
Developed as part of an Advanced AI University Project by:
* Abdul-Azeem Lotfy Abdul-Azeem
* Amr Mohamed Sayed
* Ehab Mokhtar Mohamed
* Mahmoud Ahmed Zaalouk
* Mohamed Farahat Hassan
* Youssef Gamal Hussein

**Supervised by:** Dr. Lamees Nasser & Eng. Merna
