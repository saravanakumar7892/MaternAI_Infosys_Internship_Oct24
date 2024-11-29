from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('pregnancy-risk-prediction-data-set.csv')
df_clean = df.drop(columns=["Patient ID", "Name"])
label_encoder = LabelEncoder()
df_clean['Outcome'] = label_encoder.fit_transform(df_clean['Outcome'])

X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = {column: float(request.form[column]) for column in X.columns}
        user_df = pd.DataFrame([user_input])
        prediction = model.predict(user_df)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        return render_template('predict.html', columns=X.columns, prediction=predicted_class)
    return render_template('predict.html', columns=X.columns)

if __name__ == '__main__':
    app.run(debug=True)
