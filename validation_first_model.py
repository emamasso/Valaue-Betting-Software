### Now let's perform a 10 fold validation to find the best possible model


##
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

X = df[['B365H', 'B365D','B365A', 'home_is_home',
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



loga_loss = []
macro_f1 = []


print('')
print('')
print('')
print('')
print('')
print('')
print('')
print('')
print('')
print('***********************************************************************************************************')
print('Starting now the Kfold validation')
print('***********************************************************************************************************')

for fold, (train_index, validation_index) in enumerate(Kf.split(X_train, y_train)):

    X_train_train, X_val = X_train[train_index], X_train[validation_index]
    y_train_train, y_val = y_train.iloc[train_index], y_train.iloc[validation_index]
    print('***********************************************************************************************************')
    print('Training fold number {}'.format(fold+1))
    print('***********************************************************************************************************')


    model = Sequential()

#input layer
    model.add(Input(shape=(19,)))
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

    model.fit(X_train_train, y_train_train,
               validation_data=(X_val, y_val), callbacks=[early_stop],
              epochs=150, batch_size=32, verbose=0)
    


    probs = model.predict(X_val)
    preds = np.argmax(probs, axis=1)
    f1 = f1_score(y_val, preds, average='macro')
    l_loss = log_loss(y_val, probs)
    
    macro_f1.append(f1)
    loga_loss.append(l_loss)

    print('Macro f1 score: {}'.format(f1))
    print('Logaritmic loss: {}'.format(l_loss))


print('Final values:')

print('Average macro F1 score: {}'.format(np.mean(macro_f1)))
print('Average log-loss: {}'.format(np.mean(loga_loss)))



### Looks like the model is good but non excelent