import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('spotify_preprocessed.csv')

# 1. Track Popularity by Playlist Genre
plt.figure(figsize=(10, 6))
sns.barplot(x='playlist_genre', y='track_popularity', data=df, estimator='mean')
plt.title('Average Track Popularity by Playlist Genre')
plt.xticks(rotation=45)
plt.savefig('popularity_by_genre.png')
plt.close()

# 2. Feature Distributions
numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'distribution_{feature}.png')
    plt.close()

# 3. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_matrix.png')
plt.close()

# 4. Feature vs. Genre (Box Plot for Danceability)
plt.figure(figsize=(10, 6))
sns.boxplot(x='playlist_genre', y='danceability', data=df)
plt.title('Danceability by Playlist Genre')
plt.xticks(rotation=45)
plt.savefig('danceability_by_genre.png')
plt.close()

# 5. Tempo vs. Danceability Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tempo', y='danceability', hue='playlist_genre', data=df)
plt.title('Tempo vs. Danceability by Genre')
plt.savefig('tempo_vs_danceability.png')
plt.close()

# 6. Playlist Genre Distribution
plt.figure(figsize=(8, 8))
df['playlist_genre'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Distribution of Playlist Genres')
plt.savefig('genre_distribution.png')
plt.close()