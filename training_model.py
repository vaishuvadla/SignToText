import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the processed data
images = np.load('images.npy')
labels = np.load('labels.npy')

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Define the model
classifier = RandomForestClassifier()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best estimator
best_classifier = grid_search.best_estimator_

# Train the best classifier on the entire training set
best_classifier.fit(X_train, y_train)

# Save the model and the label encoder
joblib.dump(best_classifier, 'hand_gesture_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluate the model on the test set
accuracy = best_classifier.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
