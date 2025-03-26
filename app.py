# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# ✅ Load models and encoders
models_dir = 'models'

try:
    match_model = pickle.load(open(f'{models_dir}/best_model_win.pkl', 'rb'))
    score_model = pickle.load(open(f'{models_dir}/best_model_score.pkl', 'rb'))
    toss_model = pickle.load(open(f'{models_dir}/best_model_toss.pkl', 'rb'))
    low_score_model = pickle.load(open(f'{models_dir}/best_model_low_score.pkl', 'rb'))
    player_model = pickle.load(open(f'{models_dir}/best_model_player.pkl', 'rb'))
    playoff_model = pickle.load(open(f'{models_dir}/best_model_playoffs.pkl', 'rb'))
    encoders = pickle.load(open(f'{models_dir}/encoders.pkl', 'rb'))
    scaler = pickle.load(open(f'{models_dir}/scaler.pkl', 'rb'))
    print("✅ Models and encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

app = Flask(__name__)

# 🎯 Helper Function to Encode Input
def encode_input(input_data, columns):
    for col in columns:
        if col in encoders:
            input_data[col] = encoders[col].transform([input_data[col]])[0]
    return input_data

# 🎯 API Endpoint 1: Predict Match Outcome (Toss, Venue, Opposition)
@app.route('/predict_match', methods=['POST'])
def predict_match():
    data = request.json
    input_data = {
        'toss_winner': data['toss_winner'],
        'venue': data['venue'],
        'team1': data['team1'],
        'team2': data['team2']
    }
    input_data = encode_input(input_data, input_data.keys())
    df_input = pd.DataFrame([input_data])
    prediction = match_model.predict(df_input)[0]
    predicted_winner = encoders['winner'].inverse_transform([prediction])[0]
    return jsonify({'predicted_winner': predicted_winner})

# 🎯 API Endpoint 2: Predict Match Outcome Based on Score and Powerplay
@app.route('/predict_score', methods=['POST'])
def predict_score():
    data = request.json
    input_data = np.array([[data['win_by_runs'], data['powerplay_score']]])
    prediction = score_model.predict(input_data)[0]
    predicted_winner = encoders['winner'].inverse_transform([prediction])[0]
    return jsonify({'predicted_winner': predicted_winner})

# 🎯 API Endpoint 3: Recommend Bat or Bowl Based on Venue
@app.route('/recommend_toss', methods=['POST'])
def recommend_toss():
    data = request.json
    input_data = {
        'venue': data['venue'],
        'toss_winner': data['toss_winner']
    }
    input_data = encode_input(input_data, input_data.keys())
    df_input = pd.DataFrame([input_data])
    prediction = toss_model.predict(df_input)[0]
    toss_decision = encoders['toss_decision'].inverse_transform([prediction])[0]
    return jsonify({'recommended_toss_decision': toss_decision})

# 🎯 API Endpoint 4: Predict Win Probability if RR Scores Below 160
@app.route('/predict_low_score', methods=['POST'])
def predict_low_score():
    data = request.json
    input_data = np.array([[data['win_by_runs']]])
    prediction = low_score_model.predict(input_data)[0]
    result = 'Win' if prediction == 1 else 'Loss'
    return jsonify({'win_probability_below_160': result})

# 🎯 API Endpoint 5: Predict Player of the Match
@app.route('/predict_player', methods=['POST'])
def predict_player():
    data = request.json
    input_data = {
        'venue': data['venue'],
        'team1': data['team1'],
        'team2': data['team2']
    }
    input_data = encode_input(input_data, input_data.keys())
    df_input = pd.DataFrame([input_data])
    prediction = player_model.predict(df_input)[0]
    player_name = prediction
    return jsonify({'predicted_player_of_match': player_name})

# 🎯 API Endpoint 6: Predict Playoff Qualification Based on First 7 Matches
@app.route('/predict_playoffs', methods=['POST'])
def predict_playoffs():
    data = request.json
    input_data = np.array([[data['wins_in_first_7_matches']]])
    prediction = playoff_model.predict(input_data)[0]
    result = 'Qualified' if prediction == 1 else 'Not Qualified'
    return jsonify({'playoff_qualification': result})

# 🎯 API Endpoint 7: Predict Expected Score at Chinnaswamy Stadium
@app.route('/predict_chinnaswamy_score', methods=['POST'])
def predict_chinnaswamy_score():
    data = request.json
    chinnaswamy_data = pd.DataFrame({
        'venue': ['M Chinnaswamy Stadium'],
        'team1': [data['team1']],
        'team2': [data['team2']]
    })
    chinnaswamy_data = encode_input(chinnaswamy_data.iloc[0].to_dict(), chinnaswamy_data.columns)
    df_input = pd.DataFrame([chinnaswamy_data])
    predicted_score = int(score_model.predict(df_input)[0])
    return jsonify({'expected_score_chinnaswamy': predicted_score})

# 🎯 API Endpoint 8: Predict Toss Impact on RR’s Win Probability
@app.route('/predict_toss_impact', methods=['POST'])
def predict_toss_impact():
    data = request.json
    input_data = {
        'venue': data['venue'],
        'toss_winner': data['toss_winner']
    }
    input_data = encode_input(input_data, input_data.keys())
    df_input = pd.DataFrame([input_data])
    toss_impact_prediction = toss_model.predict(df_input)[0]
    toss_impact_result = encoders['toss_decision'].inverse_transform([toss_impact_prediction])[0]
    return jsonify({'toss_impact': toss_impact_result})

# 🎯 API Endpoint 9: Predict RR’s Highest Run-Scorer in a Match
@app.route('/predict_highest_scorer', methods=['POST'])
def predict_highest_scorer():
    data = request.json
    input_data = {
        'venue': data['venue'],
        'team1': data['team1'],
        'team2': data['team2']
    }
    input_data = encode_input(input_data, input_data.keys())
    df_input = pd.DataFrame([input_data])
    prediction = player_model.predict(df_input)[0]
    highest_scorer = prediction
    return jsonify({'highest_scorer': highest_scorer})

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
