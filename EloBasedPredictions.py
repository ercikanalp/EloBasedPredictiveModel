import pandas as pd
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("CSVs/super_league.csv")


df[['Home Goals', 'Away Goals']] = df['Result'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)


BASE_ELO = 1500
K_BASE = 40
HOME_ADVANTAGE = 100


elo_ratings = defaultdict(lambda: BASE_ELO)
elo_history = []


def expected_score(rating_a, rating_b):
    """Calculate expected score for a team."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def dynamic_k_factor(home_team, away_team, home_goals, away_goals, round_info):
    """Adjust K factor dynamically based on match importance, goal difference, and competition stage."""
    K = K_BASE


    if 'important_match_condition':
        K = 60


    goal_diff = abs(home_goals - away_goals)
    if goal_diff == 1:
        multiplier = 1.0
    elif goal_diff == 2:
        multiplier = 1.5
    else:
        multiplier = (11 + goal_diff) / 8.0

    return K * multiplier


def update_elo(home_team, away_team, home_goals, away_goals, round_info):
    """Update Elo ratings for both teams after a match."""
    R_home = elo_ratings[home_team]
    R_away = elo_ratings[away_team]

    R_home_adj = R_home + HOME_ADVANTAGE

    E_home = expected_score(R_home_adj, R_away)
    E_away = expected_score(R_away, R_home_adj)

    if home_goals > away_goals:
        S_home = 1
        S_away = 0
    elif home_goals == away_goals:
        S_home = 0.5
        S_away = 0.5
    else:
        S_home = 0  # Away team wins
        S_away = 1


    adjusted_K = dynamic_k_factor(home_team, away_team, home_goals, away_goals, round_info)


    R_home_new = R_home + adjusted_K * (S_home - E_home)
    R_away_new = R_away + adjusted_K * (S_away - E_away)


    elo_ratings[home_team] = R_home_new
    elo_ratings[away_team] = R_away_new

    
    elo_history.append({
        'Round': round_info,
        'Match': f"{home_team} vs {away_team}",
        'Home Team': home_team,
        'Away Team': away_team,
        'Home Goals': home_goals,
        'Away Goals': away_goals,
        'Home Elo Before': R_home,
        'Away Elo Before': R_away,
        'Home Elo After': R_home_new,
        'Away Elo After': R_away_new
    })



for _, row in df.iterrows():
    if pd.notna(row['Home Goals']) and pd.notna(row['Away Goals']):
        round_info = row['Round Number'] if 'Round Number' in row else "?"
        update_elo(
            row['Home Team'],
            row['Away Team'],
            row['Home Goals'],
            row['Away Goals'],
            round_info
        )


final_elo = pd.DataFrame(sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True), columns=['Team', 'Elo'])


final_elo = final_elo.drop_duplicates(subset=['Team'], keep='last')

print("\n🔝 Final Elo Ratings (League + Cup, No Duplicates):")
print(final_elo)

elo_df = pd.DataFrame(elo_history)
elo_df.to_csv("elo_progression.csv", index=False)

df['Result Label'] = df.apply(
    lambda row: 1 if row['Home Goals'] > row['Away Goals'] else (-1 if row['Home Goals'] < row['Away Goals'] else 0),
    axis=1)

df['Home Elo'] = df['Home Team'].map(elo_ratings)
df['Away Elo'] = df['Away Team'].map(elo_ratings)

X = df[['Home Elo', 'Away Elo']]
y = df['Result Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print(f"\nAccuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Predict the remaining matches (if any)
remaining_matches = df[df['Round Number'] > 29]
remaining_matches['Home Elo'] = remaining_matches['Home Team'].map(elo_ratings)
remaining_matches['Away Elo'] = remaining_matches['Away Team'].map(elo_ratings)

X_remaining = remaining_matches[['Home Elo', 'Away Elo']]
predictions = model.predict(X_remaining)

remaining_matches['Predicted Result'] = predictions

print(f"\nPredictions for Remaining Matches (Round > 29):")
print(remaining_matches[['Home Team', 'Away Team', 'Predicted Result']])


points = defaultdict(int)


for _, row in df.iterrows():
    if pd.notna(row['Home Goals']) and pd.notna(row['Away Goals']):
        if row['Home Goals'] > row['Away Goals']:
            points[row['Home Team']] += 3  # Home team wins
        elif row['Home Goals'] < row['Away Goals']:
            points[row['Away Team']] += 3  # Away team wins
        else:
            points[row['Home Team']] += 1  # Draw
            points[row['Away Team']] += 1


for _, row in remaining_matches.iterrows():
    if row['Predicted Result'] == 1:
        points[row['Home Team']] += 3  # Home team predicted to win
    elif row['Predicted Result'] == -1:
        points[row['Away Team']] += 3  # Away team predicted to win
    else:
        points[row['Home Team']] += 1  # Draw
        points[row['Away Team']] += 1


final_standings = pd.DataFrame(list(points.items()), columns=['Team', 'Points'])


final_standings = final_standings.merge(final_elo, on='Team')


final_standings = final_standings.sort_values(by=['Points', 'Elo'], ascending=[False, False])

print("\nFinal Standings After Predictions (Including Total Points):")
print(final_standings)
