import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv("emails.csv")

# Step 2: Drop the 'Email No.' column
df.drop(columns=['Email No.'], inplace=True)

# Step 3: Separate features (X) and label (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column (spam label)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Step 6: Predict
y_pred = svm.predict(X_test)

# Step 7: Evaluation
print("=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Hyperparameter Tuning
params = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(kernel='linear'), params, cv=5)
grid.fit(X_train, y_train)

print("\n=== After Tuning ===")
print("Best Parameters:", grid.best_params_)
y_pred_tuned = grid.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Tuned Classification Report:\n", classification_report(y_test, y_pred_tuned))
