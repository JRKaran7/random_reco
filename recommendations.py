import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# Load preprocessed data
df = pd.read_csv("Seven_Sisters_Travel_Packages_Cleaned_Encoded.csv")

# Define features and target
X = df.drop(columns=['State'])  # Features
y = df['State']  # Target variable

# 🔹 Step 1: Feature Importance Check (RandomForest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

# Drop features with very low importance
low_importance_features = feature_importance[feature_importance < 0.01].index
X = X.drop(columns=low_importance_features)

# 🔹 Step 2: Apply PCA for Noise Reduction
pca = PCA(n_components=5)  # Keep top 5 most important components
X_pca = pca.fit_transform(X)

# Print explained variance ratio
print("PCA Explained Variance:", np.round(pca.explained_variance_ratio_, 3))

# 🔹 Step 3: Apply K-Means Clustering to create Travel Categories
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_pca)

# Update target variable (predicting Cluster instead of State)
y = df['Cluster']

# 🔹 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Step 5: Apply SMOTE for Balancing Data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 🔹 Step 6: Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 🔹 Step 7: Train Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=0.5, gamma='scale')
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=5, scoring='accuracy')
    avg_accuracy = np.mean(scores)
    
    model.fit(X_train_scaled, y_train_resampled)  # Train on resampled data
    y_pred = model.predict(X_test_scaled)  # Predict on test set
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} - Cross-Val Accuracy: {avg_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    if test_accuracy > best_accuracy:
        best_model = model
        best_accuracy = test_accuracy

# 🔹 Step 8: Save the Best Model
joblib.dump(best_model, "best_travel_recommender.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler for later use
joblib.dump(pca, "pca.pkl")  # Save PCA model
joblib.dump(kmeans, "kmeans.pkl")  # Save clustering model

print(f"Best model ({best_model.__class__.__name__}) saved with test accuracy {best_accuracy:.4f}")