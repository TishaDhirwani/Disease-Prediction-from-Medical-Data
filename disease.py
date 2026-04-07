# Disease Prediction with GUI + Graph + Confusion Matrix

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Prepare data
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

le = LabelEncoder()
y = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Accuracy
accuracy = model.score(X, y)
print("Model Accuracy:", round(accuracy, 2))

# ----------- GRAPH (Accuracy Bar) -----------
plt.figure()
plt.bar(["Random Forest"], [accuracy])
plt.title("Model Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

# ----------- CONFUSION MATRIX -----------
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ----------- GUI -----------
def predict_disease():
    input_data = []

    for entry in entries:
        val = entry.get()
        if val == "":
            val = 0
        input_data.append(int(val))

    input_array = np.array(input_data).reshape(1, -1)

    pred = model.predict(input_array)
    disease = le.inverse_transform(pred)

    result_label.config(text="Predicted Disease: " + disease[0])


# Create window
root = tk.Tk()
root.title("Disease Prediction System")
root.geometry("600x500")

tk.Label(root, text="Enter Symptoms (0 or 1)", font=("Arial", 14)).pack()

# Scrollable frame (for many symptoms)
frame = tk.Frame(root)
frame.pack()

canvas = tk.Canvas(frame, height=300)
scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Create input fields
entries = []
for col in X.columns:
    row = tk.Frame(scrollable_frame)
    row.pack(fill="x")

    tk.Label(row, text=col, width=30, anchor="w").pack(side="left")
    entry = tk.Entry(row, width=5)
    entry.pack(side="right")

    entries.append(entry)

# Predict button
tk.Button(root, text="Predict Disease", command=predict_disease).pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()