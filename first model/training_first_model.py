
### Since the nueral network looks decent let's see how it performs on the test set and what is the value betting 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, log_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

##

df = pd.read_csv('data/final_data.csv', sep=';')

mask = [df.home_goals > df.away_goals,
        df.home_goals == df.away_goals,
        df.home_goals < df.away_goals]

df['final_result'] = np.select(mask, [0, 1, 2])

y = df['final_result']

X = df[['home_is_home',
        'home_rest_days','home_total_goals','home_total_xg',
        'home_total_goals_against','home_total_xg_against','home_last_5',
        'home_PPG','away_is_home','away_rest_days',
        'away_total_goals','away_total_xg',
        'away_total_goals_against','away_total_xg_against',
        'away_last_5','away_PPG']]





scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=62, test_size=0.2)
Kf = StratifiedKFold(n_splits=10, random_state=62, shuffle=True)

model = Sequential()

#input layer
model.add(Input(shape=(16,)))
#hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))

#output layer
model.add(Dense(3, activation='softmax'))


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy')

model.fit(X_train, y_train, callbacks=[early_stop],
            epochs=150, batch_size=32, verbose=0)



probs = model.predict(X_test)

'''
preds = np.argmax(probs, axis=1)
f1 = f1_score(y_test, preds, average='macro')
l_loss = log_loss(y_test, probs)

print('Macro f1 score: {}'.format(f1))
print('Logaritmic loss: {}'.format(l_loss))
'''

### Now that we have the model, let's see how much we would have gained by betting 100 euros on the test set

initial_money = 100
current_money = initial_money
bets = 0
winnin_bets = 0

for game in range(len(y_test)):

    game_index = y_test.index[game]

    home_odd = 1/df.loc[game_index, 'B365H']
    draw_odd =  1/df.loc[game_index, 'B365D']
    away_odd =  1/df.loc[game_index, 'B365A']

    home_win_prob = probs[game][0]
    draw_prob = probs[game][1]
    away_win_prob = probs[game][2]

    home_expected_value = home_odd * home_win_prob
    draw_expected_value = draw_odd * draw_prob
    away_expected_value = away_odd * away_win_prob

    bet = 5
    if home_expected_value >= 1.05:
        current_money-=bet
        bets += 1
        if y_test.iloc[game] == 0:
            current_money += home_odd * bet
            winnin_bets += 1

    elif away_expected_value >= 1.05:
        current_money-=bet
        bets += 1
        if y_test.iloc[game] == 2:
            current_money += away_odd * bet
            winnin_bets += 1


print('Final results:')

print('Total money: {}'.format(current_money))
print('Total bets won: {}'.format(winnin_bets))
print('Total bets: {}'.format(bets))
print('Money betted: {}'.format(np.abs(100-initial_money)))
print('Revenue: {}'.format(current_money-bets))
if bets > 0:
    roi = ((current_money - initial_money) / (bets * bet)) * 100
    print(f'ROI: {roi:.2f}%')


### As said the metrics are not wonderfull and also there are absolutely no matches that would give us some advantage, so the model is worthless,
### let's try onother one