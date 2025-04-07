**⚽ Football Match Outcome Predictor**

A machine learning-based football match outcome predictor using dynamic Elo ratings and logistic regression. This system supports multiple leagues and tournaments including:

* 🇹🇷 Turkish Süper Lig
* 🇳🇱 Eredivisie
* 🇪🇸 La Liga
* 🇮🇹 Serie A
* 🏆 UEFA Champions League
* 🏅 UEFA Europa League
* 🔍 Features

**Dynamic Elo Rating System:** 
Calculates and updates team strength over time with a goal-difference-weighted Elo formula and home advantage.
Multi-League Support: Works across different CSVs representing various competitions.
Logistic Regression Model: Predicts match outcomes (home win, draw, away win) based on Elo differences.
Remaining Fixtures Simulation: Predicts outcomes of future matches and simulates full league standings.
Result Tracking: Outputs Elo rating progression and final standings with points.

Requirements
Python 3.7+
pandas
scikit-learn
matplotlib (optional, for Elo plotting)
Run the Model
python main.py
Make sure your .csv files are inside a CSVs/ folder.
