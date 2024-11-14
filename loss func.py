import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping

# Charger les données combinées
df = pd.read_csv(r'C:\Users\kedhi\Documents\combined_data_final.csv')  # Remplace par le chemin de ton fichier

# Convertir la colonne Date en datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Préparer les données pour le modèle
features = df[['Close', 'sum_sentiment', 'Volume']]  # Utiliser les colonnes 'Close', 'sum_sentiment', et 'Volume'
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Créer des séquences de données
def create_sequences(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # Inclure toutes les caractéristiques
        y.append(data[i + time_step, 0])  # Prix 'Close' à prédire
    return np.array(X), np.array(y)

# Définir le nombre de jours pour prédire
time_step = 10
X, y = create_sequences(scaled_features, time_step)

# Reshape les données pour le modèle LSTM
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire le modèle LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # prévision du prix suivant

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2) 
# Tracer l'évolution de la perte au cours des époques
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.title('Évolution de la perte au cours des époques')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()
