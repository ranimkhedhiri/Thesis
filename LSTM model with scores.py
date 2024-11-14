import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Charger les données
data = pd.read_csv(r'C:\Users\kedhi\Documents\merged_data.csv')

# Assurez-vous que la colonne 'Date' est en format datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sélectionner les colonnes pertinentes
data = data[['Date', 'Volume', 'avg_sentiment', 'Close']]

# Normaliser les colonnes sauf 'Date'
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Volume', 'avg_sentiment', 'Close']])

# Convertir les données en un DataFrame pour faciliter la manipulation
scaled_df = pd.DataFrame(scaled_data, columns=['Volume', 'avg_sentiment', 'Close'])
scaled_df['Date'] = data['Date']

# Créer des séquences pour LSTM
sequence_length = 6  # nombre de jours pour chaque séquence
X = []
y = []

for i in range(sequence_length, len(scaled_df)):
    X.append(scaled_df[['Volume', 'avg_sentiment', 'Close']].values[i-sequence_length:i])
    y.append(scaled_df['Close'].values[i])

# Convertir les données en tableaux NumPy
X, y = np.array(X), np.array(y)

# Diviser en ensemble d'entraînement et de test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construire le modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Entraîner le modèle
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Faire des prévisions
predictions = model.predict(X_test)

# Remettre les données de prédiction à l'échelle originale
predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 2)), predictions), axis=1))[:, -1]
y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 2)), y_test.reshape(-1, 1)), axis=1))[:, -1]
# Calculer le RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

# Calculer le MAPE
mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Afficher les résultats
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(y_test_actual, color='blue', label='Prix réel Bitcoin')
plt.plot(predictions, color='red', label='Prédictions LSTM')
plt.title('Bitcoin Price prediction using sentiment scores')
plt.xlabel('Temps')
plt.ylabel('Prix Bitcoin')
plt.legend()
plt.show()
