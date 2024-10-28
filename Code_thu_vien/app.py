import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Cấu hình trang
st.set_page_config(page_title="Trợ lý sức khỏe",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Thêm CSS để thay đổi màu nền của menu


# Lấy thư mục làm việc của file chính
working_dir = os.path.dirname(os.path.abspath(__file__))

# Tải các mô hình đã lưu
diabetes_model_svm = pickle.load(open(f'{working_dir}/saved_models/diabetes_model_svm.sav', 'rb'))
diabetes_model_logistic = pickle.load(open(f'{working_dir}/saved_models/diabetes_model_logictic.sav', 'rb'))

# Thanh bên để điều hướng
with st.sidebar:
    selected = option_menu('Hệ Thống Dự Đoán Bệnh',
                           ['Dự đoán bệnh tiểu đường',],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)
    

# Dự đoán bệnh tiểu đường
if selected == 'Dự đoán bệnh tiểu đường':
    # Tiêu đề trang với màu sắc cụ thể
    st.markdown("""
        <h1 style='color: #fccb90;'>Dự đoán bệnh tiểu đường bằng ML 🤖</h1>
    """, unsafe_allow_html=True)

    # Lấy dữ liệu đầu vào từ người dùng
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Số lần mang thai')

    with col2:
        Glucose = st.text_input('Mức độ glucose')

    with col3:
        BloodPressure = st.text_input('Giá trị huyết áp')

    with col1:
        SkinThickness = st.text_input('Giá trị độ dày da')

    with col2:
        Insulin = st.text_input('Mức độ insulin')

    with col3:
        BMI = st.text_input('Chỉ số BMI')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Giá trị chức năng tiểu đường')

    with col2:
        Age = st.text_input('Tuổi của người dùng')

    # Mã cho Dự đoán
    diab_diagnosis_svm = ''
    diab_diagnosis_logistic = ''
    final_diagnosis = ''
    advice_message = ''

    # Tạo nút Dự đoán
    if st.button('Kết quả kiểm tra bệnh tiểu đường'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]

        # Dự đoán từ mô hình SVM
        diab_prediction_svm = diabetes_model_svm.predict([user_input])
        if diab_prediction_svm[0] == 1:
            diab_diagnosis_svm = 'Mô hình SVM: Bạn có khả năng bị tiểu đường.'
        else:
            diab_diagnosis_svm = 'Mô hình SVM: Bạn không bị tiểu đường.'

        # Dự đoán từ mô hình Logistic Regression
        diab_prediction_logistic = diabetes_model_logistic.predict([user_input])
        if diab_prediction_logistic[0] == 1:
            diab_diagnosis_logistic = 'Mô hình Logistic Regression: Bạn có khả năng bị tiểu đường.'
        else:
            diab_diagnosis_logistic = 'Mô hình Logistic Regression: Bạn không bị tiểu đường.'

        # Hiển thị kết quả của cả hai mô hình
        st.success(diab_diagnosis_svm)
        st.success(diab_diagnosis_logistic)

        # Đưa ra kết luận chung và lời khuyên/chúc mừng
        if diab_prediction_svm[0] == 1 or diab_prediction_logistic[0] == 1:
            final_diagnosis = 'Kết luận: Bạn có khả năng bị tiểu đường.'
            advice_message = ('👨🏼‍⚕️ Hãy thăm khám bác sĩ để được tư vấn thêm về chế độ ăn uống, vận động và phương pháp điều trị.\n'
                             '🚴🏼 Bạn nên tập thể dục để kiểm soát cân nặng và giảm nguy cơ biến chứng.')
        else:
            final_diagnosis = 'Kết luận: Bạn không bị tiểu đường.'
            advice_message = '🤖 Chúc mừng! Bạn hoàn toàn khoẻ mạnh. Hãy tiếp tục duy trì lối sống lành mạnh. 🤖'

    # Hiển thị kết luận và lời khuyên/chúc mừng
    st.info(final_diagnosis)
    st.info(advice_message)
