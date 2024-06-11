#Vasileios Antonopoulos
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Load dataset
print("Loading the dataset")
dataset = pd.read_csv('log2.csv')

# Drop rows with missing values     !!!!!Did not exist in the specific dataset!!!!!
dataset = dataset.dropna()

# Separate features (X) and target variable (y)
X = dataset.drop('Action', axis=1)
y = dataset['Action']

# Print class distribution before resampling
print("\nClass distribution before resampling:")
print(y.value_counts())

# Define over and under sampling strategy
#smote = SMOTE(sampling_strategy={'reset-both': 5000}, k_neighbors=5, random_state=42)      !!!!!not working well!!!!!
over = RandomOverSampler(sampling_strategy={'reset-both': 400})
under = RandomUnderSampler(sampling_strategy={'allow': 15000})

# Define pipeline with over and under sampling
pipeline = Pipeline(steps=[('o', over), ('u', under)])

# Apply the pipeline to your data
X_resampled, y_resampled = pipeline.fit_resample(X, y)

# Print class distribution after resampling
print("\nClass distribution after resampling:")
print(y_resampled.value_counts())

#Line 52 for testing in the unprocessed  data || Line 53 for testing in the resampled data

# Split the dataset into training 70% and testing 30% sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define algorithms I'll  use to compare
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(solver='sag', multi_class='multinomial', max_iter=8500, random_state=42, n_jobs=-1),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42),
}

print("Starting training...")

# Create a dictionary to store both results and time taken for each classifier
results_and_times = {}

# Evaluate each classifier using cross-validation
for name, clf in classifiers.items():
    classifier_start_time = time.time()

    # Cross-validation for accuracy
    accuracy_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    # Cross-validation for F1 score
    f1_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    # Fit the classifier on the full training set
    clf.fit(X_train_scaled, y_train)

    # Predict probabilities on the test set for AUC score
    y_pred_proba = clf.predict_proba(X_test_scaled)
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # One Vs Rest

    # Record the end time for the current classifier
    classifier_end_time = time.time()
    classifier_elapsed_time = classifier_end_time - classifier_start_time

    # Store results and time in the dictionary
    results_and_times[name] = {
        'accuracy': mean_accuracy, 'std_dev_accuracy': std_accuracy,
        'f1_score': mean_f1, 'std_dev_f1': std_f1,
        'auc_score': auc_score, 'time': classifier_elapsed_time
    }

    print(f'{name}: Time taken: {classifier_elapsed_time:.2f} seconds')


# Displaying the results
print("=" * 80 + "\nResults and Time for Each Classifier\n")
for name, metrics in results_and_times.items():
    print(f'{name:24s} | Accuracy: {metrics["accuracy"]:.5f} (+/- {metrics["std_dev_accuracy"]:.5f}) | F1 Score: {metrics["f1_score"]:.5f} (+/- {metrics["std_dev_f1"]:.5f}) | AUC Score: {metrics["auc_score"]:.5f} | Time taken: {metrics["time"]:.2f} seconds')

print("\n" + "=" * 80 + "\nBest Classifier\n")

# Calculating a combined score for each classifier
for name, metrics in results_and_times.items():
    combined_score = (metrics['accuracy'] + metrics['f1_score'] + metrics['auc_score']) / 3
    results_and_times[name]['combined_score'] = combined_score

# Find the best classifier based on accuracy
#best_classifier = max(results_and_times, key=lambda k: results_and_times[k]['accuracy'])
#print(f'The best classifier considering accuracy is: {best_classifier}')

# Finding the best classifier based on the combined score
best_classifier = max(results_and_times, key=lambda k: results_and_times[k]['combined_score'])
print(f"The best classifier considering combined accuracy, F1, and AUC is: {best_classifier} with a score of {results_and_times[best_classifier]['combined_score']:.5f}")

print("\n" + "=" * 80 + "\nTest Accuracy\n")
print(f'Testing on Unseen Data')

# Train the best classifier on the full training set and evaluate on the test set
best_model = classifiers[best_classifier]
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f'Test Accuracy of the Best Model on Unseen Data: {test_accuracy:.5f}')

# Additional metrics. Confusion Matrix, Classification Report
print("\n" + "=" * 80 + "\nAdditional Metrics\n")
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1, digits=5))

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f'Confusion Matrix - {best_classifier}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()