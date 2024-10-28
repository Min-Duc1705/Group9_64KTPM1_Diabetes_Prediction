import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Đọc dữ liệu từ file CSV
diabetes_dataset = pd.read_csv('T:/TryHard_IT_Project/MachineLearning/Project_Disease_Prediction_Nhom9/Code_thu_vien/dataset/diabetes.csv', encoding='ISO-8859-1') 

# Tách dữ liệu và nhãn
X = diabetes_dataset.drop(columns='Outcome', axis=1).values
Y = diabetes_dataset['Outcome'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Huấn luyện mô hình Logistic Regression tự viết
model = LogisticRegressionCustom(learning_rate=0.001, n_iters=9000)  # Tăng số lần lặp
model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on Training data : ', training_data_accuracy)

# Dự đoán trên tập kiểm thử
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on Test data : ', test_data_accuracy)

# So sánh kết quả
if training_data_accuracy > test_data_accuracy:
    print("Mô hình hoạt động tốt hơn trên dữ liệu huấn luyện.")
else:
    print("Mô hình hoạt động tốt hơn trên dữ liệu kiểm tra.")

# Dự đoán cho dữ liệu đầu vào mới
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Chuyển dữ liệu đầu vào thành mảng numpy
input_data_as_numpy_array = np.asarray(input_data)

# Reshape dữ liệu để dự đoán cho một mẫu duy nhất
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Chuẩn hóa dữ liệu đầu vào sử dụng scaler đã huấn luyện
input_data_scaled = scaler.transform(input_data_reshaped)

# Dự đoán sử dụng mô hình đã huấn luyện
prediction = model.predict(input_data_scaled)  # Sửa tên từ 'classifier' thành 'model'
# prediction là một mảng với giá trị 0 hoặc 1, không cần phải chuyển đổi lại ở đây

# In kết quả dự đoán
if prediction[0] == 0:
    print('Người này không bị tiểu đường.')
else:
    print('Người này bị tiểu đường.')
