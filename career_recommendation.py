import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# AI CAREER RECOMMENDATION SYSTEM
# ==========================================

# 1. Load Dataset
df = pd.read_csv("AI_Career_Recommendation_Dataset_Balanced.csv")
print("First 5 rows:")
print(df.head())

# ==========================================
# 2. Dataset Information
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCareer Distribution:")
print(df["Career"].value_counts())

# ==========================================
# 3. Separate Features and Target
# ==========================================

X = df.drop("Career", axis=1)
y = df["Career"]

# ==========================================
# 4. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 5. Create Decision Tree Model
# ==========================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# ==========================================
# 6. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")

# ==========================================
# 7. Make Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 8. Calculate Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# ==========================================
# 9. Classification Report
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 10. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 11. Career Recommendation
# ==========================================

print("\n==============================")
print(" AI CAREER RECOMMENDATION")
print("==============================")

# Get student input
programming = int(input("Programming skill (1-10): "))
mathematics = int(input("Mathematics skill (1-10): "))
electronics = int(input("Electronics skill (1-10): "))
communication = int(input("Communication skill (1-10): "))
problem_solving = int(input("Problem-solving skill (1-10): "))
ai_interest = int(input("AI/ML interest (1-10): "))
hardware_interest = int(input("Hardware interest (1-10): "))
networking_interest = int(input("Networking interest (1-10): "))

# Create input DataFrame
student = pd.DataFrame([[
    programming,
    mathematics,
    electronics,
    communication,
    problem_solving,
    ai_interest,
    hardware_interest,
    networking_interest
]], columns=X.columns)

# Predict career
prediction = model.predict(student)[0]

# Get prediction probabilities
probabilities = model.predict_proba(student)[0]

# Get career names
career_names = model.classes_

# Display recommendation
print("\n==============================")
print(" CAREER RECOMMENDATION")
print("==============================")

print("Recommended Career:", prediction)

# Find confidence
confidence = max(probabilities) * 100

print(f"Confidence: {confidence:.2f}%")

# Display all career probabilities
print("\nCareer Probabilities:")

for career, probability in zip(career_names, probabilities):
    print(f"{career}: {probability * 100:.2f}%")
