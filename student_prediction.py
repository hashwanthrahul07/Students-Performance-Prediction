# ==========================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# Machine Learning Project using Python
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------
# STEP 1 : Student Dataset
# -------------------------------

student_data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,
                    1.5,2.2,3.8,4.8,5.8,6.8,7.8,8.8,9.8,10.5,11,12],
    "Attendance": [60,65,70,72,75,78,80,82,84,86,68,71,73,76,79,81,83,85,
                   62,67,74,77,80,82,84,86,88,90,91,93],
    "Assignments": [50,55,58,60,65,68,70,72,75,78,56,59,63,66,69,71,74,77,
                    52,57,61,64,67,70,73,76,79,82,84,86],
    "Previous_Marks": [40,45,48,52,55,58,60,64,68,72,46,50,54,57,61,65,69,73,
                       42,47,53,56,60,63,67,70,74,78,82,85],
    "Final_Marks": [42,46,50,55,60,64,68,73,78,85,48,53,58,62,66,71,76,81,
                    44,49,56,60,65,69,74,79,84,88,91,95]
}

df = pd.DataFrame(student_data)

print("\n========== STUDENT DATA ==========\n")
print(df)

# -------------------------------
# STEP 2 : Features and Target
# -------------------------------

X = df[["Study_Hours", "Attendance", "Assignments", "Previous_Marks"]]
y = df["Final_Marks"]

# -------------------------------
# STEP 3 : Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# STEP 4 : Train Model
# -------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -------------------------------
# STEP 5 : Prediction
# -------------------------------

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": np.round(y_pred,2)
})

print("\n========== ACTUAL VS PREDICTED ==========\n")
print(comparison)

# -------------------------------
# STEP 6 : Model Evaluation
# -------------------------------

print("\n========== MODEL PERFORMANCE ==========\n")

print("Intercept :", model.intercept_)

print("\nCoefficients")

for name, coef in zip(X.columns, model.coef_):
    print(f"{name:20s} : {coef:.4f}")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error :", round(mae,2))
print("Mean Squared Error  :", round(mse,2))
print("Root Mean Squared Error :", round(rmse,2))
print("R2 Score :", round(r2,4))

# -------------------------------
# STEP 7 : User Prediction
# -------------------------------

print("\n========== PREDICT STUDENT MARKS ==========\n")

hours = float(input("Enter Study Hours : "))
attendance = float(input("Enter Attendance (%) : "))
assignment = float(input("Enter Assignment Marks : "))
previous = float(input("Enter Previous Exam Marks : "))

new_student = pd.DataFrame({
    "Study_Hours":[hours],
    "Attendance":[attendance],
    "Assignments":[assignment],
    "Previous_Marks":[previous]
})

prediction = model.predict(new_student)

print("\nPredicted Final Marks :", round(prediction[0],2))

if prediction[0] >= 90:
    grade = "A+"
elif prediction[0] >= 80:
    grade = "A"
elif prediction[0] >= 70:
    grade = "B"
elif prediction[0] >= 60:
    grade = "C"
elif prediction[0] >= 50:
    grade = "D"
else:
    grade = "F"

print("Predicted Grade :", grade)

# -------------------------------
# STEP 8 : Visualization
# -------------------------------

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title("Actual vs Predicted Student Marks")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(X.columns, model.coef_)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.grid(axis='y')
plt.show()