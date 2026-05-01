# 🎬 Netflix Insight Engine

A complete end-to-end Data Science web application analyzing 8,807 Netflix 
titles using Python, Streamlit, Machine Learning, NLP, AI and Voice Recognition.

![Python](https://img.shields.io/badge/Python-3.13.3-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

👉 [Click here to open the live app](https://your-app-name.streamlit.app)

---

## 📋 Project Overview

The Netflix Insight Engine is a professional-grade Data Science application 
built as an MCA final year project. It demonstrates the complete Data Science 
pipeline — from raw data exploration and cleaning to machine learning, 
natural language processing, AI integration and deployment.

---

## 🧩 Modules

| Module | Feature | Technology |
|--------|---------|------------|
| M1 | Data Preview and EDA | Pandas, Streamlit |
| M2 | Smart Search and Filter | Pandas, Streamlit |
| M3 | EDA Visualizations | Plotly Express |
| M4 | ML Recommendation System | TF-IDF, Cosine Similarity |
| M5 | Sentiment Analysis | VADER NLP |
| M6 | Genre Predictor | Logistic Regression |
| M7 | Word Cloud | WordCloud, NLTK |
| M8 | Country Heatmap | Plotly Choropleth |
| M9 | Content Growth Timeline | Plotly Animation |
| M10 | AI Chatbot | Google Gemini API |
| M11 | Voice Recognition | SpeechRecognition |

---

## 🛠️ Tech Stack

- **Language** — Python 3.13.3
- **Framework** — Streamlit
- **Data** — Pandas, NumPy
- **Visualization** — Plotly, Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn (TF-IDF, Cosine Similarity, Logistic Regression)
- **NLP** — VADER Sentiment, NLTK, WordCloud
- **AI** — Google Gemini API (free tier)
- **Voice** — SpeechRecognition, PyAudio
- **Deployment** — Streamlit Cloud, GitHub

---

## 📁 Project Structure
netflix-insight-engine/
│
├── app.py                          # Home page
├── requirements.txt                # All dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Files excluded from GitHub
├── netflix_titles.csv              # Dataset (Kaggle)
│
├── pages/
│   ├── 1_Data_Preview.py
│   ├── 2_Smart_Search.py
│   ├── 3_EDA_Visualizations.py
│   ├── 4_Recommendations.py
│   ├── 5_Sentiment_Analysis.py
│   ├── 6_Genre_Predictor.py
│   ├── 7_Word_Cloud.py
│   ├── 8_Country_Heatmap.py
│   ├── 9_Content_Timeline.py
│   ├── 10_AI_Chatbot.py
│   └── 11_Voice_Recognition.py
│
├── utils/
│   ├── data_loader.py
│   ├── nlp_helpers.py
│   ├── ml_models.py
│   └── voice_helper.py
│
└── assets/
└── style.css
---

## ⚙️ How to Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/vikeshgohil/netflix-insight-engine.git
cd netflix-insight-engine
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Download the dataset**

Download `netflix_titles.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) 
and place it in the root project folder.

**Step 4 — Add your Gemini API key**

Create a `.env` file in the root folder and add:

Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

**Step 5 — Run the app**
```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source** — [Netflix Movies and TV Shows — Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **Author** — Shivam Bansal
- **Rows** — 8,807 titles
- **Columns** — 12 original + 4 derived
- **Years covered** — 2008 to 2021

---

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| Sentiment — Positive titles | 59.5% |
| Sentiment — Negative titles | 7.7% |
| Genre predictor top-3 accuracy | 70–75% |
| Peak Netflix growth year | 2019 — 1,960 titles |
| Countries represented | 127 |

---

## 👤 Author

**Vikesh Gohil**
MCA Final Year Student
📧 your-email@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/vikesh-gohil-538262218)
🐙 [GitHub](https://github.com/vikeshgohil)

---

## 📄 License

This project is licensed under the MIT License.