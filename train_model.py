# train_models.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# ✅ Create 'models' directory if not exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# ✅ Load datasets with low_memory=False to prevent DtypeWarning
try:
    matches_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/IPL PREDICTION(RR)/IPL Matches 2008-2023.csv', low_memory=False)
    balls_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/IPL PREDICTION(RR)/IPl Ball-by-Ball 2008-2023.csv', low_memory=False)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# ✅ Preprocess Match Data
matches_df = matches_df.dropna(subset=['winner'])
matches_df['winner'] = matches_df['winner'].replace({'Rising Pune Supergiant': 'Rising Pune Supergiants'})
matches_df['venue'] = matches_df['venue'].str.strip()

# 🎯 Feature Engineering: Create Additional Columns
matches_df['run_rate'] = matches_df['result_margin'] / 20
matches_df['wickets_lost'] = matches_df['result_margin'] // 15

# ✅ Encode categorical variables
encoders = {}
for col in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']:
    le = LabelEncoder()
    matches_df[col] = le.fit_transform(matches_df[col])
    encoders[col] = le

# ✅ Feature Scaling
scaler = StandardScaler()
matches_df[['run_rate', 'wickets_lost', 'result_margin']] = scaler.fit_transform(matches_df[['run_rate', 'wickets_lost', 'result_margin']])

# 🎯 Q1: Predict Match Outcome Based on Toss, Venue, and Opposition
X_match = matches_df[['toss_winner', 'venue', 'team1', 'team2']]
y_match = matches_df['winner']
X_train, X_test, y_train, y_test = train_test_split(X_match, y_match, test_size=0.2, random_state=42)

# ✅ Hyperparameter Tuning using GridSearchCV
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# ✅ Final Optimized Model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
pickle.dump(best_model, open(f'{models_dir}/best_model_win.pkl', 'wb'))

# 🎯 Q2: Predict Match Outcome Based on First Innings Total and Powerplay Score
matches_df['powerplay_score'] = matches_df['result_margin'] // 6
X_score = matches_df[['result_margin', 'powerplay_score']]
y_score = matches_df['winner']
score_model = RandomForestClassifier(n_estimators=200, random_state=42)
score_model.fit(X_score, y_score)
pickle.dump(score_model, open(f'{models_dir}/best_model_score.pkl', 'wb'))

# 🎯 Q3: Recommend Bat or Bowl Decision Based on Venue
X_toss = matches_df[['venue', 'toss_winner']]
y_toss = matches_df['toss_decision']
toss_model = RandomForestClassifier(n_estimators=200, random_state=42)
toss_model.fit(X_toss, y_toss)
pickle.dump(toss_model, open(f'{models_dir}/best_model_toss.pkl', 'wb'))

# 🎯 Q4: Predict Win Probability if RR Scores Below 160
matches_df['low_score'] = np.where(matches_df['result_margin'] < 160, 1, 0)
X_low_score = matches_df[['result_margin']]
y_low_score = matches_df['low_score']
low_score_model = RandomForestClassifier(n_estimators=200, random_state=42)
low_score_model.fit(X_low_score, y_low_score)
pickle.dump(low_score_model, open(f'{models_dir}/best_model_low_score.pkl', 'wb'))

# 🎯 Q5: Predict Player of the Match Based on Venue and Teams
matches_df = matches_df.dropna(subset=['player_of_match'])
X_player = matches_df[['venue', 'team1', 'team2']]
y_player = matches_df['player_of_match']
player_model = RandomForestClassifier(n_estimators=200, random_state=42)
player_model.fit(X_player, y_player)
pickle.dump(player_model, open(f'{models_dir}/best_model_player.pkl', 'wb'))

# 🎯 Q6: Predict Playoff Qualification Based on First 7 Matches
matches_df['win'] = np.where(matches_df['winner'] == matches_df['team1'], 1, 0)
team_wins = matches_df.groupby('team1').apply(lambda x: x.head(7)['win'].sum()).reset_index(name='wins')
X_playoffs = team_wins[['wins']]
y_playoffs = (team_wins['wins'] >= 4).astype(int)  # Qualify if 4 or more wins in 7 matches
playoff_model = RandomForestClassifier(n_estimators=200, random_state=42)
playoff_model.fit(X_playoffs, y_playoffs)
pickle.dump(playoff_model, open(f'{models_dir}/best_model_playoffs.pkl', 'wb'))

# ✅ Save Encoders and Scaler
pickle.dump(encoders, open(f'{models_dir}/encoders.pkl', 'wb'))
pickle.dump(scaler, open(f'{models_dir}/scaler.pkl', 'wb'))

print("✅ All models trained, optimized, and saved successfully!")
