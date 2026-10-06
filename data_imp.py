import soccerdata as sd
import pandas as pd
import time 

## let's start by getting all the games played in the season from 21-22 season
## in the european top 5 leagues

leagues = ['ESP-La Liga', 'ITA-Serie A', 'ENG-Premier League', 'GER-Bundesliga', 'FRA-Ligue 1']
seasons = ['2021/2022', '2022/2023', '2023/2024', '2024/2025', '2025/2026']

games_final = []

for league in leagues:
    games = sd.Understat(league, seasons)

    games_final.append(games.read_schedule())
    print('{} inserito con successo'.format(league))
    time.sleep(15)


games_df = pd.concat(games_final, axis=0)
games_df = games_df.reset_index()

games_df.to_csv('data/games.csv', index=False, sep=';')




#### And now let's import all the odds
'''
odds_final = []

for league in leagues: 
    for season in seasons:
        odds = sd.MatchHistory(leagues=league, seasons=season, proxy='tor)

        odds_final.append(odds.read_games())
        print('{} {} succesfully saved'.format(league, season))
        time.sleep(5)


odds_df = pd.concat(odds_final, axis=0)
odds_df = odds_df.reset_index()

odds_df.to_csv('data/odds.csv', index=False, sep=';')

'''

### The previous code is the fastest way possible to collect data, but it may not work, 
## try manually downloading data from https://www.football-data.co.uk/ then execute the following code:

directory = Path('C:/Users/emanu/OneDrive/Desktop/progetti/value_betting_software/data/tmp') # write your own path

df_list = []

for file in directory.glob('*.csv'):
    df = pd.read_csv(file)
    df_list.append(df)

odds_df = pd.concat(df_list, axis=0, ignore_index=True)
odds_df = odds_df.drop_duplicates()
odds_df.to_csv('data/odds.csv', index=False)