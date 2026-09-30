# ⚽ Football Player Performance Indicator

A football analytics web application developed to analyze and understand football player performance using data analysis and machine learning.

The application provides player analysis, player comparison, player replacement recommendations, Expected Goals (xG) prediction, and player position prediction through an interactive web interface.

## 🚀 Features

- ⚽ Football player performance analysis
- 📊 Player statistics exploration
- 🔄 Head-to-head player comparison
- 🔍 Player replacement recommendation
- 🤖 Expected Goals (xG) prediction
- 🏷️ Player position prediction
- 🌍 FIFA World Cup themed user interface
- 📈 Interactive web-based dashboard

## 🛠️ Technologies Used

- Python
- Flask
- HTML
- CSS
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Git & GitHub

## 🤖 Machine Learning

The application uses trained machine learning models for different prediction tasks.

### 1. Expected Goals (xG) Prediction

A regression model is used to predict the Expected Goals (xG) value of a football player based on player statistics.

Model file:

`xg_model_bundle.pkl`

### 2. Position Prediction

A classification model is used to predict the player's playing position based on relevant player statistics.

Model files:

`position_classifier.pkl`

`knn_position_bundle.pkl`

### 3. Player Replacement Recommendation

The application provides player replacement recommendations based on player performance characteristics and statistical similarity.

## 📂 Project Structure

```text
Football_Player_Performance_Indicator/
│
├── app.py
├── requirements.txt
├── README.md
├── world_cup_player_stats.csv
│
├── xg_model_bundle.pkl
├── position_classifier.pkl
├── knn_position_bundle.pkl
│
├── static/
│   ├── fifa_world_cup.glb
│   ├── loading.mp4
│   └── style.css
│
├── templates/
│   └── index.html
│
└── .vscode/
    └── settings.json


    ## 📊 Dataset

The project uses football player statistics stored in:

`world_cup_player_stats.csv`

The dataset contains player performance-related features that are used for analysis and machine learning predictions.

## 🎯 Project Objective

The main objective of the Football Player Performance Indicator is to provide a data-driven platform for analyzing football players and generating useful insights from player statistics.

The application combines data analysis, machine learning, and web development to create an interactive football analytics platform.

## 🔮 Future Scope

- Integration of live football data
- Real-time player performance analysis
- Advanced player recommendation systems
- Improved machine learning models
- Cloud deployment
- Integration with additional football leagues and competitions

## 👩‍💻 Project Type

**Data Science and Analytics Project**

### Developed Using

Python | Flask | Machine Learning | HTML | CSS | Pandas | Scikit-learn