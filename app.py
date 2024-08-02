from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            df = pd.read_csv(file)
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            plt.figure()
            plt.hist(y_test, bins=20, alpha=0.5, label='Actual')
            plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted')
            plt.legend()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_str = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close()
            
            return render_template('index.html', accuracy=accuracy, img_data=img_str)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)