import pandas as pd
import pickle
import joblib
import numpy as np
from data_engineering import *

data = df_new
#data = pd.read_csv('prediction_data/final_data.csv', sep = ';')

data['home_is_home'] = 1 
data['away_is_home'] = 0

data_to_predict = data[['B365H', 'B365D', 'B365A', 
       'home_is_home', 'home_rest_days', 'home_total_goals',
       'home_total_xg', 'home_total_goals_against', 'home_total_xg_against',
       'home_last_5', 'home_PPG', 
       'away_is_home', 'away_rest_days', 'away_total_goals', 'away_total_xg',
       'away_total_goals_against', 'away_total_xg_against', 'away_last_5',
       'away_PPG', 'last_5_difference', 'PPG_difference', 
       'home_elo', 'away_elo', 'elo_difference']]





with open('model_v3.pkl', 'rb') as file:
    model = pickle.load(file)


predictions = list(model.predict(data_to_predict))
probabilities =list(model.predict_proba(data_to_predict))

games = []

for i in range(data.shape[0]):
    games.append('-'.join((data['home'].loc[i], data['away'].loc[i])))



dictionary = {'Game':games, 'Forecasted result':predictions}


final_data_frame = pd.DataFrame(dictionary)


mask = [final_data_frame['Forecasted result'] == 0,
        final_data_frame['Forecasted result'] == 1,
        final_data_frame['Forecasted result'] == 2]

final_data_frame['Forecasted result'] = np.select(mask, ['1', 'X', '2'], default='N/A')


prob = [max(x) for x in probabilities]

final_data_frame['Probability'] = prob


odds = pd.read_csv('prediction_data/games_odds.csv', sep = ';')

data_with_odds = pd.merge(data, odds[['id', 'H', 'D', 'A']], on='id', how='left')

final_data_frame['Home Win'] = data_with_odds['H']
final_data_frame['Draw'] = data_with_odds['D']
final_data_frame['Away Win'] = data_with_odds['A']

quote_cols = ['Away Win', 'Draw', 'Home Win']

quote_scelte = [final_data_frame['Home Win'],
    final_data_frame['Draw'],
    final_data_frame['Away Win']]

final_data_frame['Bet'] = np.select(mask, quote_scelte, default=np.nan)

final_data_frame['Expected Value'] = final_data_frame['Probability'] * final_data_frame['Bet']