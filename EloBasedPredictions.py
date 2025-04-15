import pandas as pd
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox

BASE_ELO = 1500
K = 40
HOME_ADVANTAGE = 100

# Function to compute expected score
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# Function to calculate goal difference multiplier
def goal_difference_multiplier(goal_diff):
    if goal_diff == 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0

# Function to update elo ratings
def update_elo(elo_ratings, home_team, away_team, home_goals, away_goals):
    R_home = elo_ratings[home_team]
    R_away = elo_ratings[away_team]

    R_home_adj = R_home + HOME_ADVANTAGE
    E_home = expected_score(R_home_adj, R_away)
    E_away = expected_score(R_away, R_home_adj)

    if home_goals > away_goals:
        S_home, S_away = 1, 0
    elif home_goals == away_goals:
        S_home = S_away = 0.5
    else:
        S_home, S_away = 0, 1

    multiplier = goal_difference_multiplier(abs(home_goals - away_goals))
    K_adj = K * multiplier

    R_home_new = R_home + K_adj * (S_home - E_home)
    R_away_new = R_away + K_adj * (S_away - E_away)

    elo_ratings[home_team] = R_home_new
    elo_ratings[away_team] = R_away_new

# Load league data and return unique teams
def load_league_teams(file):
    df = pd.read_csv(f"CSVs/{file}")
    return sorted(set(df['Home Team']).union(df['Away Team']))

# Recalculate elo from match history
def calculate_elo_from_file(files):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    for file in files:
        df = pd.read_csv(f"CSVs/{file}")
        df[['Home Goals', 'Away Goals']] = df['Result'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)
        for _, row in df.iterrows():
            if pd.notna(row['Home Goals']) and pd.notna(row['Away Goals']):
                update_elo(elo_ratings, row['Home Team'], row['Away Team'], row['Home Goals'], row['Away Goals'])
    return elo_ratings

# Predict outcome of match
def predict_outcome():
    home_team = home_team_var.get()
    away_team = away_team_var.get()
    home_league = home_league_var.get()
    away_league = away_league_var.get()
    europe_home = european_var_home.get()
    europe_away = european_var_away.get()

    if not all([home_team, away_team, home_league, away_league]):
        messagebox.showerror("Error", "Please select both teams and their leagues.")
        return

    files = list(set([home_league, away_league] + ([europe_home] if europe_home else []) + ([europe_away] if europe_away else [])))
    ratings = calculate_elo_from_file(files)

    home_elo = ratings[home_team] + HOME_ADVANTAGE
    away_elo = ratings[away_team]

    proba_home = expected_score(home_elo, away_elo)
    proba_away = 1 - proba_home

    if proba_home > 0.55:
        result = f"Prediction: {home_team} will likely win."
    elif proba_home < 0.45:
        result = f"Prediction: {away_team} will likely win."
    else:
        result = "Prediction: It's likely to be a draw."

    result += f"\nWin Probability: {home_team} {proba_home:.2%} | {away_team} {proba_away:.2%}"
    messagebox.showinfo("Prediction", result)

# Update teams dropdown based on selected league
def update_teams(var, menu, league_file):
    teams = load_league_teams(league_file.get())
    menu['menu'].delete(0, 'end')
    for team in teams:
        menu['menu'].add_command(label=team, command=tk._setit(var, team))
    var.set(teams[0])


root = tk.Tk()
root.title("Elo Match Predictor")
root.geometry("600x400")

league_files = ["super_league.csv", "eredivisie.csv", "la_liga.csv", "serie_a.csv", "bundesliga.csv", "epl.csv", "ligue_1.csv"]
european_files = ["", "ucl.csv", "uel.csv", "uecl.csv"]

home_league_var = tk.StringVar(value=league_files[0])
away_league_var = tk.StringVar(value=league_files[0])

home_team_var = tk.StringVar()
away_team_var = tk.StringVar()
european_var_home = tk.StringVar()
european_var_away = tk.StringVar()

# Home team
ttk.Label(root, text="Home League").pack()
home_league_menu = ttk.OptionMenu(root, home_league_var, league_files[0], *league_files, command=lambda _: update_teams(home_team_var, home_team_menu, home_league_var))
home_league_menu.pack()

ttk.Label(root, text="Home Team").pack()
home_team_menu = ttk.OptionMenu(root, home_team_var, "")
home_team_menu.pack()
update_teams(home_team_var, home_team_menu, home_league_var)

# Away team
ttk.Label(root, text="Away League").pack()
away_league_menu = ttk.OptionMenu(root, away_league_var, league_files[0], *league_files, command=lambda _: update_teams(away_team_var, away_team_menu, away_league_var))
away_league_menu.pack()

ttk.Label(root, text="Away Team").pack()
away_team_menu = ttk.OptionMenu(root, away_team_var, "")
away_team_menu.pack()
update_teams(away_team_var, away_team_menu, away_league_var)

# European competitions (optional)
ttk.Label(root, text="Home Team European Competition").pack()
europe_menu_home = ttk.OptionMenu(root, european_var_home, european_files[0], *european_files)
europe_menu_home.pack()

ttk.Label(root, text="Away Team European Competition").pack()
europe_menu_away = ttk.OptionMenu(root, european_var_away, european_files[0], *european_files)
europe_menu_away.pack()

# Predict button
ttk.Button(root, text="Predict Match Outcome", command=predict_outcome).pack(pady=10)

root.mainloop()