import pandas as pd


games = pd.read_csv('data/games.csv', sep=';')
odds = pd.read_csv('data/odds.csv', sep=';')


odds_filtered = odds.loc[:, ['date', 'game', 'home_team','away_team', 'B365H', 'B365D',	'B365A', 'FTR']]

games_filtered = games.loc[:, ['date', 'game', 'home_team','away_team','home_golas', 'away_goals', 'home_xg', 'away_xg']]

print(odds_filtered.shape)
print(games_filtered.shape)
