import os
import time
import requests
import pandas as pd
import soccerdata as sd


key = '1a110a3c654411bdfa1bbb3d1e436a6c3b83d0a982e8e484f9b5b43ef16a4208'


bookmaker = 'Bet365'

api_leagues = ['italy-serie-a', 'england-premier-league', 'spain-laliga', 'germany-bundesliga', 'france-ligue-1']


games = []
for league in api_leagues:
    response = requests.get('https://api.odds-api.io/v3/events', 
                            params={'apiKey': key, 'sport': 'football', 'limit' : 10, 'league' : league})
    events = response.json()


    for game in events:
        if game['status'] == 'pending':
                match_id = game['id']
                home = game['home']
                away = game['away']
                date = game['date']

                games.append([match_id, date, home, away])

    print(f'{league} data correctly downloaded')

games_df = pd.DataFrame(games, columns=['id', 'date', 'home', 'away'])



odds = []
for game in games: 
    response = requests.get('https://api.odds-api.io/v3/odds', 
                            params={'apiKey': key, 'sport': 'football', 'eventId': game[0], 'bookmakers': bookmaker})
    
    match = response.json()

    match_id = match.get('id')
    
    try:
        odds_data = match['bookmakers']['Bet365'][0]['odds'][0]
        odd_h = float(odds_data['home'])
        odd_d = float(odds_data['draw'])
        odd_a = float(odds_data['away'])
        
    except (KeyError, IndexError):
        continue
    odds.append([match_id, odd_h, odd_d, odd_a])
            


odds_df = pd.DataFrame(odds, columns=['id', 'H', 'D', 'A'])


games_odds = pd.merge(games_df, odds_df, on = 'id')

home_names_odds = odds_df['HomeTeam'].unique()
away_names_odds = odds_df['AwayTeam'].unique()

api_names = pd.concat([home_names_odds, away_names_odds], axis=0).unique()

leagues = ['ESP-La Liga', 'ITA-Serie A', 'ENG-Premier League', 'GER-Bundesliga', 'FRA-Ligue 1']
season =  '2026/2027'

games_final = []
for league in leagues:
    games = sd.Understat(league, season)

    games_final.append(games.read_schedule())
    print('{} succesfully downloaded'.format(league))
    time.sleep(15)


matches_df = pd.concat(games_final, axis=0)
matches_df = matches_df.reset_index()

home_names = matches_df['home_team'].unique()
away_names = matches_df['away_team'].unique()

understat_names = pd.concat([home_names, away_names], axis=0).unique()


print('Undestat names')
print('')
print(understat_names)
print('')
print('Odds api names')
print('')
print(api_names)