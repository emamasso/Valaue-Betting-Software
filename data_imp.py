import soccerdata as sd
import pandas as pd
import time 

## let's start by getting all the games played in the season from 21-22 season
## in the european top 5 leagues

leagues = ['ESP-La Liga', 'ITA-Serie A', 'ENG-Premier League', 'GER-Bundesliga', 'FRA-Ligue 1']
seasons = ['2021/2022', '2022/2023', '2023/2024', '2024/2025', '2025/2026']
'''
games_final = []

for league in leagues:
    games = sd.Understat(league, seasons)

    games_final.append(games.read_schedule())
    print('{} inserito con successo'.format(league))
    time.sleep(15)


games_df = pd.concat(games_final, axis=0)
games_df = games_df.reset_index()

games_df.to_csv('data/games.csv', index=False, sep=';')


'''

#### And now let's import all the odds

odds_final = []

for league in leagues: 
    for season in seasons:
        odds = sd.MatchHistory(leagues=league, seasons=season)

        odds_final.append(odds.read_games())
        print('{} {} inserito con successo'.format(league, season))
        time.sleep(5)


odds_df = pd.concat(odds_final, axis=0)
odds_df = odds_df.reset_index()

odds_df.to_csv('data/odds.csv', index=False, sep=';')