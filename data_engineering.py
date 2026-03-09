## In order to get the final web app we need to get all the data of the upcoming matches

##########################################
import pandas as pd
import requests
import soccerdata as sd
import time
import numpy as np
##########################################


## At first let's collect all the upcoming matches and theuir odds from the API-Football


key = '1a110a3c654411bdfa1bbb3d1e436a6c3b83d0a982e8e484f9b5b43ef16a4208'


bookmaker = 'Bet365'

api_leagues = ['italy-serie-a', 'england-premier-league', 'spain-laliga', 'germany-bundesliga', 'france-ligue-1']


print('**********************************************************************************************************')
print("                                  Starting collecting odds' data")
print('**********************************************************************************************************')

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


games_odds.to_csv('prediction_data/games_odds.csv', sep = ';', index=False)

print('**********************************************************************************************************')
print(f'           Correctly obtained data of {games_odds.shape[0]} games of this round')
print('**********************************************************************************************************')



#### And now let's collect all the other data about teams
print('**********************************************************************************************************')
print("                                Starting collecting all the teams' staitstics")
print('**********************************************************************************************************')


leagues = ['ESP-La Liga', 'ITA-Serie A', 'ENG-Premier League', 'GER-Bundesliga', 'FRA-Ligue 1']
season =  '2025/2026'

games_final = []
for league in leagues:
    games = sd.Understat(league, season)

    games_final.append(games.read_schedule())
    print('{} succesfully downloaded'.format(league))
    time.sleep(15)


matches_df = pd.concat(games_final, axis=0)
matches_df = matches_df.reset_index()
matches_filtered = matches_df.loc[:, ['season', 'date', 'game', 'home_team','away_team','home_goals', 'away_goals', 'home_xg', 'away_xg']]

matches_filtered.to_csv('prediction_data/matches_data.csv', index=False, sep=';')

print('**********************************************************************************************************')
print("                              Done, please wait just a few more moments!")
print('**********************************************************************************************************')


#### Now it's just needed to perform some data engeneering so that we can get all the feature of the training data

stakes = ['H', 'D', 'A']

for stake in stakes:
    games_odds[stake] = 1/games_odds[stake]

prov_sum = games_odds.A + games_odds.D + games_odds.H

for stake in stakes:
    games_odds[stake] = games_odds[stake]/prov_sum

df_home = matches_filtered[['season', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg']].copy()


df_home.columns = ['season', 'date', 'team', 'opponent', 'goals_for', 'goals_against', 'xG_for', 'xG_against']

df_home['is_home'] = 1


df_away = matches_filtered[['season', 'date', 'away_team', 'home_team', 'away_goals', 'home_goals', 'away_xg', 'home_xg']].copy()

df_away.columns = ['season', 'date', 'team', 'opponent', 'goals_for', 'goals_against', 'xG_for', 'xG_against']


df_away['is_home'] = 0



df_long = pd.concat([df_home, df_away], axis=0)

df_long['date'] = pd.to_datetime(df_long['date']).dt.normalize()


## And now lets's order the dataframe by team and date and compute all the feature that could be usefull:
## Rest days since lst match, total goals scored and conceded, total xG of the team and conceded, total points in the last 5 matches and
## points per game

df_long = df_long.sort_values(by=['team', 'date']).reset_index(drop=True)


mask = [(df_long.goals_for > df_long.goals_against).fillna(False).to_numpy(dtype=bool), 
        (df_long.goals_for == df_long.goals_against).fillna(False).to_numpy(dtype=bool),
        (df_long.goals_for < df_long.goals_against).fillna(False).to_numpy(dtype=bool)]

df_long['points gained'] = np.select(mask, [3, 1, 0], default=np.nan)

groups = df_long.groupby(['season', 'team'])

df_long['rest_days'] = groups['date'].diff().dt.days

df_long['total_goals'] = groups['goals_for'].transform(lambda x: x.cumsum())

df_long['total_xg'] = groups['xG_for'].transform(lambda x: x.cumsum())

df_long['total_goals_against'] = groups['goals_against'].transform(lambda x: x.cumsum())

df_long['total_xg_against'] = groups['xG_against'].transform(lambda x: x.cumsum())

df_long['last_5'] = groups['points gained'].transform(lambda x: x.rolling(5).sum())

df_long['match_played'] = groups.cumcount()

df_long['PPG'] = groups['points gained'].transform(lambda x: x.cumsum())/df_long['match_played']




############ Now it's time to merge this new features with the new ones


df_played = df_long.dropna(subset=['goals_for', 'goals_against']).copy()

df_played = df_played.sort_values(by=['team', 'date'])

last_games = df_played.drop_duplicates(subset=['team'], keep='last').reset_index(drop=True)


last_games_2 = last_games.copy()

df = pd.merge(last_games, last_games_2, left_on=['team'], right_on=['opponent'])
df.sort_values(by = 'team_x')


df_filtered = df[['date_x', 'team_x',
                    'total_goals_x', 'total_xg_x', 
                    'total_goals_against_x', 'total_xg_against_x', 
                    'last_5_x', 'match_played_x', 'PPG_x', 'rest_days_x']]


team_mapping = {
    
    'AC Milan': 'AC Milan',
    'Atalanta': 'Atalanta BC',
    'Bologna': 'Bologna FC',
    'Cagliari': 'Cagliari Calcio',
    'Como': 'Como 1907',
    'Cremonese': 'US Cremonese',
    'Fiorentina': 'ACF Fiorentina',
    'Genoa': 'Genoa CFC',
    'Inter': 'Inter Milano',
    'Juventus': 'Juventus Turin',
    'Lazio': 'Lazio Rome',
    'Lecce': 'US Lecce',
    'Napoli': 'SSC Napoli',
    'Parma Calcio 1913': 'Parma Calcio',
    'Pisa': 'Pisa SC',
    'Roma': 'AS Roma',
    'Sassuolo': 'Sassuolo Calcio',
    'Torino': 'Torino FC',
    'Udinese': 'Udinese Calcio',
    'Verona': 'Hellas Verona',

    
    'Arsenal': 'Arsenal FC',
    'Aston Villa': 'Aston Villa',
    'Bournemouth': 'AFC Bournemouth',
    'Brentford': 'Brentford', 
    'Brighton': 'Brighton & Hove Albion',
    'Burnley': 'Burnley FC',
    'Chelsea': 'Chelsea FC',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton FC',
    'Fulham': 'Fulham FC',
    'Leeds': 'Leeds United',
    'Liverpool': 'Liverpool FC',
    'Manchester City': 'Manchester City',
    'Manchester United': 'Manchester United',
    'Newcastle United': 'Newcastle United',
    'Nottingham Forest': 'Nottingham Forest',
    'Sunderland': 'Sunderland AFC',
    'Tottenham': 'Tottenham Hotspur',
    'West Ham': 'West Ham United',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers', 

   
    'Alaves': 'Deportivo Alaves',
    'Athletic Club': 'Athletic Bilbao',
    'Atletico Madrid': 'Atletico Madrid',
    'Barcelona': 'FC Barcelona',
    'Celta Vigo': 'RC Celta de Vigo',
    'Elche': 'Elche CF',
    'Espanyol': 'Espanyol Barcelona',
    'Getafe': 'Getafe CF',
    'Girona': 'Girona FC',
    'Levante': 'Levante UD',
    'Mallorca': 'RCD Mallorca',
    'Osasuna': 'CA Osasuna',
    'Rayo Vallecano': 'Rayo Vallecano',
    'Real Betis': 'Real Betis Seville',
    'Real Madrid': 'Real Madrid',
    'Real Oviedo': 'Real Oviedo',
    'Real Sociedad': 'Real Sociedad San Sebastian',
    'Sevilla': 'Sevilla FC',
    'Valencia': 'Valencia CF',
    'Villarreal': 'Villarreal CF',

    
    'Augsburg': 'FC Augsburg',
    'Bayer Leverkusen': 'Bayer Leverkusen',
    'Bayern Munich': 'Bayern Munich',
    'Borussia Dortmund': 'Borussia Dortmund',
    'Borussia M.Gladbach': 'Borussia Monchengladbach',
    'Eintracht Frankfurt': 'Eintracht Frankfurt',
    'FC Cologne': '1. FC Cologne',
    'FC Heidenheim': '1. FC Heidenheim',
    'Freiburg': 'SC Freiburg',
    'Hamburger SV': 'Hamburger SV',
    'Hoffenheim': 'TSG Hoffenheim',
    'Mainz 05': 'FSV Mainz',
    'RasenBallsport Leipzig': 'RB Leipzig',
    'St. Pauli': 'FC St. Pauli',
    'Union Berlin': 'Union Berlin',
    'VfB Stuttgart': 'VfB Stuttgart',
    'Werder Bremen': 'Werder Bremen',
    'Wolfsburg': 'VFL Wolfsburg',

   
    'Angers': 'Angers SCO',
    'Auxerre': 'AJ Auxerre',
    'Brest': 'Stade Brest 29',
    'Le Havre': 'Le Havre AC',
    'Lens': 'Racing Club De Lens',
    'Lille': 'Lille OSC',
    'Lorient': 'FC Lorient',
    'Lyon': 'Olympique Lyon',
    'Marseille': 'Olympique Marseille',
    'Metz': 'FC Metz',
    'Monaco': 'AS Monaco',
    'Nantes': 'FC Nantes',
    'Nice': 'OGC Nice',
    'Paris FC': 'Paris FC',
    'Paris Saint Germain': 'Paris Saint-Germain',
    'Rennes': 'Stade Rennais FC',
    'Strasbourg': 'Strasbourg Alsace',
    'Toulouse': 'Toulouse FC'
}

df_filtered['team_x'] = df_filtered['team_x'].replace(team_mapping)

df_new = pd.merge(games_odds, df_filtered, left_on=['home'], right_on=['team_x'])

df_new = pd.merge(df_new, df_filtered, left_on=['away'], right_on=['team_x'])

df_new.columns = ['id', 'date', 'home', 'away', 
                  
                  'B365H', 'B365D', 'B365A',

                  'date_x_x', 'team_x_x',

                  'home_total_goals','home_total_xg', 'home_total_goals_against',
                  'home_total_xg_against','home_last_5', 'home_matches_played', 'home_PPG', 'home_rest_days',

                  'date_x_y', 'team_x_y',
                   
                  'away_total_goals','away_total_xg',
                  'away_total_goals_against','away_total_xg_against',
                  'away_last_5', 'away_matches_played', 'away_PPG', 'away_rest_days']

df_new = df_new.dropna()

df_new = df_new.reset_index(drop=True)

df_new.to_csv('prediction_data/final_data.csv', sep=';', index=False)


print('**********************************************************************************************************')
print('              All the final data of the {} games correctly downloaded'.format(df_new.shape[0]))
print('**********************************************************************************************************')