# COMMAND: py -m streamlit run two.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Load the student data
@st.cache_data
def load_student_data():
    return pd.read_csv("student-data.csv")

all_data = load_student_data()

all_subject_codes = ["BS112","ES101","ES109","ES112","ES116","HS101","2BS01","2HS01","2IT01","2IT02","2IT03","2IT04","2HS02","21T05","21T06","21T07","21T08","21T09","OE-1","3HS01","3IT01","3IT02","3IT03","3IT04","3IT05","OE-2","3IT06","3IT07","3IT31","PE-1","3IT43"]
all_result_codes = ["SPI-1", "CPI-1", "SPI-2", "CPI-2", "SPI-3", "SPI-4", "CPI-4", "SPI-5", "CPI-5", "SPI-6", "CPI-6"]

only_subject_data = all_data.drop(columns=all_result_codes)
only_result_data = all_data.drop(columns=all_subject_codes)

# Count occurrences of 'FP' for each student
fp_counts = all_data.apply(lambda row: row.value_counts().get('FP', 0), axis=1)

# Maximum allowed FP is 5
maximum_fp = 5

# Mask to identify students with more than 5 'FP'
mask = (fp_counts <= maximum_fp)

# Remove failed students
passed_students_data = all_data[mask]

grade_to_mark = {
    "AA": 10,
    "AB": 9,
    "BB": 8,
    "BC": 7,
    "CC": 6,
    "CD": 5,
    "DD": 4,
    "NA": 0,
    "FP": 0,
    "PP": 7
}

passed_students_numeric_data = passed_students_data.replace(grade_to_mark)
all_numeric_data = all_data.replace(grade_to_mark)

# Create a Streamlit UI
st.title("Student Data Analysis")

# Sidebar to choose the analysis option
analysis_option = st.sidebar.selectbox("Choose Analysis", ["Subject-Wise Grades", "Student-Wise Grades", "Failed Students", "SPI Prediction"])

# Function to make pie chart for subject's grades
def subject_wise_grade_chart(subject_code):
    try:
        value_counts = all_data[subject_code].value_counts()
        st.write(f'Pie Chart for {subject_code}')
        st.pyplot(pie_chart(value_counts))
        st.table(all_data[subject_code])
    except IndexError:
        st.write(f'No subject with given ID: {subject_code}.')
    except Exception as e:
        st.write(f'No subject with given ID: {subject_code}.')

# Function to make pie chart for student's grades
def student_wise_grade_chart(student_id):
    try:
        student_index = only_subject_data.index[only_subject_data['ID No.'] == student_id].tolist()[0]
        student_data_show = only_subject_data.iloc[student_index]
        student_data = student_data_show.drop("ID No.")
        student_data = student_data.drop("SPI-7")
        value_counts = student_data.value_counts()
        st.write(f'Grade Distribution for Student {student_id}')
        st.pyplot(pie_chart(value_counts))
        st.table(student_data_show)

    except IndexError:
        st.write(f'Student with ID {student_id} not found.')
    except Exception as e:
        st.write(f'Student with ID {student_id} not found.')


# Function to perform linear regression
def perform_linear_regression(student_data):
    train_data = pd.DataFrame(student_data[:-1], columns=['Data1'])

    # Create the second DataFrame with the nth element
    df2 = pd.DataFrame([student_data[-1]], columns=['Data2'])

    X_train = train_data.index.values.reshape(-1,1)
    y_train = train_data

    X_test = np.array([48]).reshape(-1,1)
    y_test = np.array([y_train.mean()])

    # Fitting Linear regression model
    # Simplest regression technique
    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction_linear = model.predict(X_test)
    print(prediction_linear)
    return round(prediction_linear[0][0],2)

# Function to perform Support Vector Regression (SVR)
def perform_svr(student_data):
    train_data = pd.DataFrame(student_data[:-1], columns=['Data1'])

    # Create the second DataFrame with the nth element
    df2 = pd.DataFrame([student_data[-1]], columns=['Data2'])

    X_train = train_data.index.values.reshape(-1,1)
    y_train = train_data

    X_test = np.array([48]).reshape(-1,1)
    y_test = np.array([y_train.mean()])

    # Fitting support vector regression model
    # kernal technique, similar to SVM
    model_svm_regression = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

    model_svm_regression.fit(X_train, y_train)

    prediction_svm = model_svm_regression.predict(X_test)

    return round(prediction_svm[0],2)

# Function to perform Gaussian Process Regression
def perform_gaussian_process(student_data):
    train_data = pd.DataFrame(student_data[:-1], columns=['Data1'])

    # Create the second DataFrame with the nth element
    df2 = pd.DataFrame([student_data[-1]], columns=['Data2'])

    X_train = train_data.index.values.reshape(-1,1)
    y_train = train_data

    X_test = np.array([48]).reshape(-1,1)
    y_test = np.array([y_train.mean()])

    # Fitting Gaussian Process regression model
    # Bell Like structure, stores data in a curve manner
    # Higher the curve, harder the prediction
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    gp.fit(X_train, y_train)

    prediction_gaussian, sigma = gp.predict(X_test, return_std=True)

    return round(prediction_gaussian[0],2)

# Function to perform Ridge Regression
def perform_ridge_regression(student_data):
    train_data = pd.DataFrame(student_data[:-1], columns=['Data1'])

    # Create the second DataFrame with the nth element
    df2 = pd.DataFrame([student_data[-1]], columns=['Data2'])

    X_train = train_data.index.values.reshape(-1,1)
    y_train = train_data

    X_test = np.array([48]).reshape(-1,1)
    y_test = np.array([y_train.mean()])

    # ridge regression model fit
    # Ridge regression is like linear regression, but with an extra twist to prevent overfitting. 
    alpha = 1.0
    ridge = Ridge(alpha=alpha)

    ridge.fit(X_train, y_train)
    prediction_ridge = ridge.predict(X_test)
    return round(prediction_ridge[0][0],2)

# Function to perform Lasso Regression
def perform_lasso_regression(student_data):
    train_data = pd.DataFrame(student_data[:-1], columns=['Data1'])

    # Create the second DataFrame with the nth element
    df2 = pd.DataFrame([student_data[-1]], columns=['Data2'])

    X_train = train_data.index.values.reshape(-1,1)
    y_train = train_data

    X_test = np.array([48]).reshape(-1,1)
    y_test = np.array([y_train.mean()])

    alpha=1.0
    # Lasso regression model fit
    # lasso eliminates less important features, similar to Linear regression
    lasso = Lasso(alpha=alpha)

    lasso.fit(X_train, y_train)

    prediction_lasso = lasso.predict(X_test)

    return round(prediction_lasso[0],2)

# Function to perform Lasso Regression
def perform_lstm_regression(student_data):
    data = np.asarray(student_data).astype('float32')

    # Define the number of previous time steps to consider for prediction
    look_back = 3

    # Prepare the data in sequences with corresponding target values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])

    X, y = np.array(X), np.array(y)

    # Reshape the data for LSTM input (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Create an LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=100, batch_size=1)

    # Make predictions
    test_input = np.array([student_data[-2], student_data[-3], student_data[-4]])
    test_input = np.asarray(test_input).astype('float32')
    test_input = np.reshape(test_input, (1, 1, look_back))
    predicted_output = model.predict(test_input)
    return round(predicted_output[0][0],2)


# Function to remove failed students
def remove_failed_students(max_fp):
    fp_counts = all_data.apply(lambda row: row.value_counts().get('FP', 0), axis=1)
    mask = (fp_counts <= max_fp)
    passed_students_data = all_data[mask]
    return passed_students_data

# Function to create a pie chart
def pie_chart(value_counts):
    fig, ax = plt.subplots()
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    return fig


if analysis_option == "Subject-Wise Grades":
    st.header("Subject-Wise Grades")
    subject_code_input = st.text_input("Enter a subject code:", "BS112")
    subject_wise_grade_chart(subject_code_input)

elif analysis_option == "Student-Wise Grades":
    st.header("Student-Wise Grades")
    student_id_input = st.text_input("Enter a student's ID:", "20IT422")
    student_wise_grade_chart(student_id_input)

elif analysis_option == "Failed Students":
    st.header("Failed Students")
    max_fp = st.slider("Maximum Allowed FP:", 0, 10, 5)
    passed_students_data = remove_failed_students(max_fp)
    st.write("Number of passed students: ", passed_students_data.shape[0])

elif analysis_option == "SPI Prediction":
    st.header("SPI Prediction")

    student_id_input = st.text_input("Enter a student's ID:", "20IT422")
    try:
        student_index =  all_numeric_data.index[all_numeric_data['ID No.'] == student_id_input].tolist()[0]
        student_data_show = all_numeric_data.iloc[student_index]
        student_data = student_data_show.drop("ID No.")

        student_data = student_data.values
        
        st.subheader("Linear Regression Model")
        predicted_output_linear = perform_linear_regression(student_data)
        st.write("Predicted SPI (Linear Regression):", predicted_output_linear)

        st.subheader("Support Vector Regression Model")
        predicted_output_svm = perform_svr(student_data)
        st.write("Predicted SPI (Support Vector Regression):", predicted_output_svm)

        st.subheader("Gaussian Process Regression Model")
        predicted_output_gaussian = perform_gaussian_process(student_data)
        st.write("Predicted SPI (Gaussian Process Regression):", predicted_output_gaussian)

        st.subheader("Ridge Regression Model")
        predicted_output_ridge = perform_ridge_regression(student_data)
        st.write("Predicted SPI (Ridge Regression):", predicted_output_ridge)

        st.subheader("Lasso Regression Model")
        predicted_output_lasso = perform_lasso_regression(student_data)
        st.write("Predicted SPI (Lasso Regression):", predicted_output_lasso)
        
        st.subheader("LSTM Prediction Model")
        predicted_output_lasso = perform_lstm_regression(student_data)
        st.write("Predicted SPI (LSTM Prediction):", predicted_output_lasso)

        st.table(student_data_show)


    except IndexError:
        st.write(f'Student with ID {student_id_input} not found.')
    except Exception as e:
        st.write(f'Student with ID {student_id_input} not found.')


# Run the Streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.text("Machine Learning Project: Group G1")
st.sidebar.text("Student Grade prediction model")
st.sidebar.text("")
st.sidebar.text("Students:")
st.sidebar.text("20IT407: Jainam Vipulkumar Patel")
st.sidebar.text("20IT422: Manav Amishkumar Shah")
st.sidebar.text("20IT448: Kunal Alpeshbhai Panchal")
st.sidebar.text("20IT481: Arya Nimeshkumar Shah")
st.sidebar.text("")
st.sidebar.text("Data source: student-data.csv")