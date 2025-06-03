import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import uuid

# Load the dataset
df = pd.read_csv('spotify_dataset.csv')

# 1. Inspect the data
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 2. Handle missing values (if any)
df = df.dropna()  # Drop rows with missing values (if applicable)

# 3. Remove duplicates based on track_id
df = df.drop_duplicates(subset=['track_id'], keep='first')

# 4. Select relevant numerical features for clustering
numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[numerical_features]

# 5. Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_features)

# 6. Encode categorical features
le_genre = LabelEncoder()
le_subgenre = LabelEncoder()
df['genre_encoded'] = le_genre.fit_transform(df['playlist_genre'])
df['subgenre_encoded'] = le_subgenre.fit_transform(df['playlist_subgenre'])

# Save preprocessed data
df.to_csv('spotify_preprocessed.csv', index=False)
print("Preprocessed data saved to 'spotify_preprocessed.csv'")