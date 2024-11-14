import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lire le fichier Excel
file_path = r'C:\Users\kedhi\Documents\posts6.xlsx'
df = pd.read_excel(file_path)

# Convertir la colonne 'created' en format datetime
df['created'] = pd.to_datetime(df['created'], format='%d/%m/%Y')

# Vérifier et remplir les valeurs manquantes
df['title'] = df['title'].fillna('')
df['title'] = df['title'].astype(str)

# Initialiser VADER
analyzer = SentimentIntensityAnalyzer()

# Fonction d'analyse de sentiment
def sentiment_analysis(title):
    score = analyzer.polarity_scores(title)
    return score['compound']

# Appliquer l'analyse de sentiment
df['sentiment_score'] = df['title'].apply(sentiment_analysis)

# Calculer la moyenne des scores de sentiment par jour
daily_sentiment_avg = df.groupby(df['created'].dt.date)['sentiment_score'].mean().reset_index(name='avg_sentiment')

# Afficher les résultats
print(daily_sentiment_avg)

# Sauvegarder dans un fichier CSV
output_path = r'C:\Users\kedhi\Documents\daily_sentiment_avg.csv'
daily_sentiment_avg.to_csv(output_path, index=False)

print(daily_sentiment_avg)








