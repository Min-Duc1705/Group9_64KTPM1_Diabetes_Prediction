import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Định nghĩa lớp SimpleSVM
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.05, n_iters=1000):
        self.learning_rate = learning_rate  # Tốc độ học
        self.lambda_param = lambda_param  # Tham số điều chỉnh
        self.n_iters = n_iters  # Số lần lặp
        self.w = None  # Trọng số ban đầu
        self.b = None  # Bias ban đầu

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Số mẫu và số đặc trưng
        y_ = np.where(y <= 0, -1, 1)  # Thay đổi nhãn thành -1 và 1 nếu cần

        # Khởi tạo vector trọng số và bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

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

# Huấn luyện mô hình SVM tự viết
classifier = SimpleSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
classifier.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện
X_train_prediction = classifier.predict(X_train)
X_train_prediction = np.where(X_train_prediction <= 0, 0, 1)  # Chuyển đổi nhãn lại thành 0 và 1

# Tính toán độ chính xác trên tập huấn luyện
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Điểm độ chính xác của dữ liệu huấn luyện: ', training_data_accuracy)

# Dự đoán trên tập kiểm thử
X_test_prediction = classifier.predict(X_test)
X_test_prediction = np.where(X_test_prediction <= 0, 0, 1)

# Tính toán độ chính xác trên tập kiểm thử
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Điểm độ chính xác của dữ liệu kiểm tra: ', test_data_accuracy)

# So sánh kết quả
if training_data_accuracy > test_data_accuracy:
    print("Mô hình hoạt động tốt hơn trên dữ liệu huấn luyện.")
else:
    print("Mô hình hoạt động tốt hơn trên dữ liệu kiểm tra.")

# Dự đoán cho dữ liệu đầu vào mới
input_data = (5,166,72,19,175,25.8,0.587,51)

# Chuyển dữ liệu đầu vào thành mảng numpy
input_data_as_numpy_array = np.asarray(input_data)

# Reshape dữ liệu để dự đoán cho một mẫu duy nhất
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Chuẩn hóa dữ liệu đầu vào sử dụng scaler đã huấn luyện
input_data_scaled = scaler.transform(input_data_reshaped)

# Dự đoán sử dụng mô hình đã huấn luyện
prediction = classifier.predict(input_data_scaled)
prediction = np.where(prediction <= 0, 0, 1)

# In kết quả dự đoán
if prediction[0] == 0:
    print('Người này không bị tiểu đường.')
else:
    print('Người này bị tiểu đường.')
