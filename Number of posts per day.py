import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
file_path = r'C:\Users\kedhi\Documents\posts6.xlsx'
df = pd.read_excel(file_path)

# Convertir la colonne 'created' en format datetime si nécessaire
df['created'] = pd.to_datetime(df['created'], errors='coerce')

# Extraire la date sans l'heure
df['created'] = df['created'].dt.date

# Grouper par date et compter les titres
titles_per_day = df.groupby('created').size()

# Afficher les résultats
print(titles_per_day.head())  # Afficher les 5 premiers résultats

# Afficher l'histogramme
plt.figure(figsize=(12, 6))

# Ploter l'histogramme
titles_per_day.plot(kind='bar', color='skyblue')

# Ajouter un titre et des labels
plt.title('Number of posts per Day', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('number of posts', fontsize=12)

# Sélectionner un sous-ensemble de dates pour l'axe X (par exemple, chaque 7 jours)
ticks_to_show = titles_per_day.index[::7]  # Prendre chaque 7e date

# Afficher uniquement ces dates sur l'axe X
plt.xticks(ticks=range(0, len(titles_per_day), 7), labels=ticks_to_show, rotation=45, ha='right')

# Ajuster la mise en page
plt.tight_layout()

# Ajouter une grille sur l'axe Y
plt.grid(axis='y')

# Afficher le graphique
plt.show()
