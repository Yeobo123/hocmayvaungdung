from flask import Flask, render_template
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

app = Flask(__name__)

@app.route('/')
def index():
    # Tải dữ liệu và xây dựng mô hình
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Xây dựng mô hình KNN
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Tính toán các chỉ số
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    return render_template('index.html', accuracy=accuracy, recall=recall, precision=precision)

if __name__ == '__main__':
    app.run(debug=True)
