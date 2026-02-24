import pandas as pd
import numpy as np

### In this section the data is merged in a single dataset and some preprocessing will be performed

games = pd.read_csv('data/games.csv', sep=';')
odds = pd.read_csv('data/odds.csv', sep=';')


## First let's select only the features that are needed

odds_filtered = odds.loc[:, ['Date', 'HomeTeam','AwayTeam', 'B365H', 'B365D',	'B365A', 'FTR']]

games_filtered = games.loc[:, ['season', 'date', 'game', 'home_team','away_team','home_goals', 'away_goals', 'home_xg', 'away_xg']]


## Since the teams' names are not the same in the two df, let's address this problem 

games_teams = sorted(games_filtered['home_team'].unique())

odds_teams = sorted(odds_filtered['HomeTeam'].unique())

### The following dictionary contains all the different names, so we can replace them in the dataaframe
different_names_dict = {
    'AC Milan': 'Milan',
    'Arminia Bielefeld': 'Bielefeld',
    'Athletic Club': 'Ath Bilbao',
    'Atletico Madrid': 'Ath Madrid',
    'Bayer Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund',
    "Borussia M.Gladbach": "M'gladbach",
    'Celta Vigo': 'Celta',
    'Clermont Foot': 'Clermont',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'Espanyol': 'Espanol',
    'FC Cologne': 'FC Koln',
    'FC Heidenheim': 'Heidenheim',
    'Greuther Fuerth': 'Greuther Furth',
    'Hamburger SV': 'Hamburg',
    'Hertha Berlin': 'Hertha',
    'Mainz 05': 'Mainz',
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle',
    "Nottingham Forest": "Nott'm Forest",
    'Paris Saint Germain': 'Paris SG',
    'Parma Calcio 1913': 'Parma',
    'RasenBallsport Leipzig': 'RB Leipzig',
    'Rayo Vallecano': 'Vallecano',
    'Real Betis': 'Betis',
    'Real Oviedo': 'Oviedo',
    'Real Sociedad': 'Sociedad',
    'Real Valladolid': 'Valladolid',
    'Saint-Etienne': 'St Etienne',
    'St. Pauli': 'St Pauli',
    'VfB Stuttgart': 'Stuttgart',
    'Wolverhampton Wanderers': 'Wolves'
    }


games_filtered['home_team'] = games_filtered['home_team'].replace(different_names_dict)
games_filtered['away_team'] = games_filtered['away_team'].replace(different_names_dict)



## Let's aslso address the problem of the dates that are different in the 2 df
games_filtered['date'] = pd.to_datetime(games_filtered['date']).dt.normalize()

odds_filtered['Date'] = pd.to_datetime(odds_filtered['Date'], dayfirst=True).dt.normalize()


## And now finally merge the data in a single df
final_df = pd.merge(games_filtered, odds_filtered, left_on=['date', 'home_team','away_team'], right_on=['Date', 'HomeTeam','AwayTeam'])



### Now that the data has been cleared it's time for some feature engeneering to obtain all the final feature we want to include







#################################################################################################################################################

############################################################# FEATURE ENGENEERING ###############################################################

#################################################################################################################################################



## First of all let's get the probability of every outcome (1/odd)

stakes = ['B365H', 'B365D', 'B365A']

for stake in stakes:
    final_df[stake] = 1/final_df[stake]

prov_sum = final_df.B365A + final_df.B365D + final_df.B365H

for stake in stakes:
    final_df[stake] = final_df[stake]/prov_sum


## Now we need to double every game so that we can work properly 
df_home = final_df[['season', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg']].copy()


df_home.columns = ['season', 'date', 'team', 'opponent', 'goals_for', 'goals_against', 'xG_for', 'xG_against']

df_home['is_home'] = 1


df_away = final_df[['season', 'date', 'away_team', 'home_team', 'away_goals', 'home_goals', 'away_xg', 'home_xg']].copy()

df_away.columns = ['season', 'date', 'team', 'opponent', 'goals_for', 'goals_against', 'xG_for', 'xG_against']


df_away['is_home'] = 0



df_long = pd.concat([df_home, df_away], axis=0)


## And now lets's order the dataframe by team and date and compute all the feature that could be usefull:
## Rest days since lst match, total goals scored and conceded, total xG of the team and conceded, total points in the last 5 matches and
## points per game

df_long = df_long.sort_values(by=['team', 'date']).reset_index(drop=True)


mask = [df_long.goals_for > df_long.goals_against, 
        df_long.goals_for == df_long.goals_against,
        df_long.goals_for < df_long.goals_against]

df_long['points gained'] = np.select(mask, [3, 1, 0])

groups = df_long.groupby(['season', 'team'])

df_long['rest_days'] = groups['date'].diff().dt.days

df_long['total_goals'] = groups['goals_for'].transform(lambda x: x.cumsum().shift(1))

df_long['total_xg'] = groups['xG_for'].transform(lambda x: x.cumsum().shift(1))

df_long['total_goals_against'] = groups['goals_against'].transform(lambda x: x.cumsum().shift(1))

df_long['total_xg_against'] = groups['xG_against'].transform(lambda x: x.cumsum().shift(1))

df_long['last_5'] = groups['points gained'].transform(lambda x: x.shift(1).rolling(5).sum())

df_long['match_played'] = groups.cumcount()

df_long['PPG'] = groups['points gained'].transform(lambda x: x.cumsum())/df_long['match_played']




############ Now it's time to merge this new features with the new ones

df = pd.merge(final_df, df_long, left_on=['date', 'home_team'], right_on=['date', 'team']).rename(columns={'goals_for':'home_goals_for',
                                                                                                           	'goals_against':'home_goals_against',
                                                                                                            'xG_for':'home_xG_for',	
                                                                                                            'xG_against':'home_xG_against'	,
                                                                                                            'is_home':'home_is_home',	
                                                                                                            'points gained':'home_points_gained',	
                                                                                                            'rest_days':'home_rest_days',	
                                                                                                            'total_goals':'home_total_goals',	
                                                                                                            'total_xg':'home_total_xg',	
                                                                                                            'total_goals_against':'home_total_goals_against',	
                                                                                                            'total_xg_against':'home_total_xg_against',	
                                                                                                            'last_5':'home_last_5',	
                                                                                                            'match_played':'home_match_played',	
                                                                                                            'PPG':'home_PPG'})


df = pd.merge(df, df_long, left_on=['date', 'away_team'], right_on=['date', 'team']).rename(columns={'goals_for':'away_goals_for',
                                                                                                           	'goals_against':'away_goals_against',
                                                                                                            'xG_for':'away_xG_for',	
                                                                                                            'xG_against':'away_xG_against'	,
                                                                                                            'is_home':'away_is_home',	
                                                                                                            'points gained':'away_points_gained',	
                                                                                                            'rest_days':'away_rest_days',	
                                                                                                            'total_goals':'away_total_goals',	
                                                                                                            'total_xg':'away_total_xg',	
                                                                                                            'total_goals_against':'away_total_goals_against',	
                                                                                                            'total_xg_against':'away_total_xg_against',	
                                                                                                            'last_5':'away_last_5',	
                                                                                                            'match_played':'away_match_played',	
                                                                                                            'PPG':'away_PPG'})


df = df.dropna()

df = df.reset_index(drop=True)


df.to_csv('data/final_data.csv', sep=';', index=False)

print('Data frame with shape {} correctly saved as CSV'.format(df.shape))
