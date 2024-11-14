import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Charger les données
merged_data = pd.read_csv(r'C:\Users\kedhi\Documents\merged_data_filtered.csv')

# Nettoyer les colonnes
merged_data.columns = merged_data.columns.str.strip()

# Créer la colonne de la différence de 'Close' si ce n'est pas déjà fait
merged_data['Close_diff'] = merged_data['Close'].diff()

# Supprimer les valeurs NaN après la différence
merged_data.dropna(inplace=True)

# Sélectionner les colonnes pertinentes pour le modèle VAR
data_for_var = merged_data[['avg_sentiment', 'Close_diff']]

# Vérifier si les séries sont stationnaires avec le test ADF (Augmented Dickey-Fuller)
for col in data_for_var.columns:
    adf_test = adfuller(data_for_var[col])
    print(f"{col}: p-value = {adf_test[1]}")

# Si les p-values sont > 0.05, cela signifie que les séries ne sont pas stationnaires
# et qu'il faut les différencier davantage (par exemple en prenant les premières différences)

# Appliquer le modèle VAR après avoir stationnarisé les données
model = VAR(data_for_var)
# Choisir le nombre de lags optimal en utilisant le critère AIC
lag_order = model.select_order(maxlags=15).aic
print(f"Optimal number of lags: {lag_order}")

# Ajuster le modèle VAR avec le nombre de lags choisi
var_model = model.fit(lag_order)

# Résumé des résultats
print(var_model.summary())

# Faire des prédictions
forecast_steps = 10  # Par exemple, prévoir les 10 prochaines valeurs
forecast = var_model.forecast(data_for_var.values[-lag_order:], steps=forecast_steps)

# Afficher les prévisions
forecast_df = pd.DataFrame(forecast, columns=data_for_var.columns)
print(forecast_df)

# Visualiser les prévisions
plt.figure(figsize=(10, 6))
plt.plot(merged_data['Date'], data_for_var['avg_sentiment'], label='Avg Sentiment')
plt.plot(merged_data['Date'], data_for_var['Close_diff'], label='Close Diff')
plt.legend(loc='best')
plt.title('Sentiment vs Close Diff (after VAR model)')
plt.show()

