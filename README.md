!pip install gradio pandas scikit-learn

import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Split data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1500)
model.fit(X_train, y_train)

# Accuracy print
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Recommendation function
def recommendation(pred, glucose, bmi):
    if pred == 1:
        text = "High Risk\n\n"
        if glucose > 130:
            text += "• Reduce sugar intake\n"
        if bmi > 28:
            text += "• Try to lower weight\n"
        text += "• Exercise daily and avoid junk food"
    else:
        text = "Low Risk\n\n• Continue healthy lifestyle"
    return text


# Main prediction function
def predict(Glucose, BloodPressure, Insulin, BMI, Age):

    avg = data.mean()

    user_input = [[
        avg["Pregnancies"],
        Glucose,
        BloodPressure,
        avg["SkinThickness"],
        Insulin,
        BMI,
        avg["DiabetesPedigreeFunction"],
        Age
    ]]

    pred = model.predict(user_input)[0]
    result = "Positive" if pred == 1 else "Negative"

    return f"Result: {result}\n\n{recommendation(pred, Glucose, BMI)}"


# Gradio UI
ui = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Simple Diabetes Checker (5 Inputs)",
    description="Enter your values to check diabetes risk."
)

# IMPORTANT FOR COLAB
ui.launch(debug=True, share=True)
