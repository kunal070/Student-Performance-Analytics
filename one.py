
# import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score , classification_report
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# read data from csv file
all_data = pd.read_csv("student-data.csv")

all_data = all_data.fillna(0)

all_subject_codes = ["BS112","ES101","ES109","ES112","ES116","HS101","2BS01","2HS01","2IT01","2IT02","2IT03","2IT04","2HS02","21T05","21T06","21T07","21T08","21T09","OE-1","3HS01","3IT01","3IT02","3IT03","3IT04","3IT05","OE-2","3IT06","3IT07","3IT31","PE-1","3IT43"]
all_result_codes = ["SPI-1", "CPI-1", "SPI-2", "CPI-2", "SPI-3", "SPI-4", "CPI-4", "SPI-5", "CPI-5", "SPI-6", "CPI-6"]

only_subject_data = all_data.drop(columns=all_result_codes)
only_result_data = all_data.drop(columns=all_subject_codes)

# Total number of rows is equal to total number of students
print("Total Number of Students: ", all_data.shape[0])

# here 11 columns for CPI & SPI (sem 3 CPI is not present in data) & 1 Column for ID Number
print("Total Subjects: ", all_data.shape[1] - 12)

# head shows how data is stored
print("\n\n")
print(all_data.head())

# function to make pie chart for subject's grades
def subject_wise_grade_chart(subject_code):
    # fect count of grades for specific subject code
    value_counts = all_data[subject_code].value_counts()

    # Plot a pie chart for subject's grades
    plt.figure(figsize=(8, 8))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart for {subject_code}')
    plt.show()

# sample test for a subject
# subject_code_input = input("Enter a subject code: ")
# subject_wise_grade_chart(subject_code_input)

# function to make pie chart for student's grades
def student_wise_grade_chart(student_id):
    # get student's rows
    student_index =  only_subject_data.index[only_subject_data['ID No.'] == student_id].tolist()[0]
    student_data = only_subject_data.iloc[student_index]
    student_data = student_data.drop("ID No.")

    # count grades from student's row
    value_counts = student_data.value_counts()

    # Plot a pie chart for the specific student's grade distribution
    plt.figure(figsize=(8, 8))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Grade Distribution for Student {student_id}')
    plt.show()

# sample test for a student
# student_id_input = input("Enter student's ID: ")
student_id_input = "20IT422"
student_wise_grade_chart(student_id_input)

# Count occurrences of 'FP' for each student
fp_counts = all_data.apply(lambda row: row.value_counts().get('FP', 0), axis=1)

# Maximum allowed FP is 5
maximum_fp = 5

# Mask to identify students with more than 5 'FP'
mask = (fp_counts <= maximum_fp)

# Remove failed students
passed_students_data = all_data[mask]

# Display the passed students
print("\n\nNumber of passed students: ", passed_students_data.shape[0])
print("\n\nPASSED STUDENTS DATA:")
print(passed_students_data.head(10))

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

print("\n\nPassed students numeric data:")
print(passed_students_numeric_data.head(10))


student_index =  passed_students_numeric_data.index[passed_students_numeric_data['ID No.'] == student_id_input].tolist()[0]
student_data = passed_students_numeric_data.iloc[student_index]
print(f"\n\nStudent {student_id_input}'s Data: ")
print(student_data)
student_data = student_data.drop("ID No.")
student_data = student_data.values
print(student_data)

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

# Fitting support vector regression model
# kernal technique, similar to SVM
model_svm_regression = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Fitting Gaussian Process regression model
# Bell Like structure, stores data in a curve manner
# Higher the curve, harder the prediction
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# ridge regression model fit
# Ridge regression is like linear regression, but with an extra twist to prevent overfitting. 
alpha = 1.0
ridge = Ridge(alpha=alpha)

# Lasso regression model fit
# lasso eliminates less important features, similar to Linear regression
lasso = Lasso(alpha=alpha)


model.fit(X_train, y_train)
model_svm_regression.fit(X_train, y_train)
gp.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)


# # semester wise X and Y Data
# x = passed_students_numeric_data.iloc[:, [1,2,3,4,5,6]].values
# y = passed_students_numeric_data.iloc[:, [7]].values

# sem_two_x = passed_students_numeric_data.iloc[:, [9,10,11,12,13,14]].values
# sem_two_y = passed_students_numeric_data.iloc[:, [15]].values

# sem_three_x = passed_students_numeric_data.iloc[:, [17,18,19,20,21,22]].values
# sem_three_y = passed_students_numeric_data.iloc[:, [23]].values

# sem_four_x = passed_students_numeric_data.iloc[:, [24,25,26,27,28,29]].values
# sem_four_y = passed_students_numeric_data.iloc[:, [30]].values

# sem_five_x = passed_students_numeric_data.iloc[:, [32,33,34,35,36,37,38]].values
# sem_five_y = passed_students_numeric_data.iloc[:, [39]].values

# sem_six_x = passed_students_numeric_data.iloc[:, [41,42,43,44,45,46]].values
# sem_six_y = passed_students_numeric_data.iloc[:, [47]].values


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# model_svm_regression.fit(X_train, y_train)
# gp.fit(X_train, y_train)
# ridge.fit(X_train, y_train)
# lasso.fit(X_train, y_train)

# X_train, X_test, y_train, y_test = train_test_split(sem_two_x, sem_two_y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# model_svm_regression.fit(X_train, y_train)
# gp.fit(X_train, y_train)
# ridge.fit(X_train, y_train)
# lasso.fit(X_train, y_train)

# X_train, X_test, y_train, y_test = train_test_split(sem_three_x, sem_three_y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# model_svm_regression.fit(X_train, y_train)
# gp.fit(X_train, y_train)
# ridge.fit(X_train, y_train)
# lasso.fit(X_train, y_train)


# X_train, X_test, y_train, y_test = train_test_split(sem_four_x, sem_four_y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# model_svm_regression.fit(X_train, y_train)
# gp.fit(X_train, y_train)
# ridge.fit(X_train, y_train)
# lasso.fit(X_train, y_train)

# X_train, X_test, y_train, y_test = train_test_split(sem_five_x, sem_five_y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# model_svm_regression.fit(X_train, y_train)
# gp.fit(X_train, y_train)
# ridge.fit(X_train, y_train)
# lasso.fit(X_train, y_train)

# X_train, X_test, y_train, y_test = train_test_split(sem_six_x, sem_six_y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# model_svm_regression.fit(X_train, y_train)
# gp.fit(X_train, y_train)
# ridge.fit(X_train, y_train)
# lasso.fit(X_train, y_train)


# prediction on testing data
prediction_linear = model.predict(X_test)
prediction_svm = model_svm_regression.predict(X_test)
prediction_gaussian, sigma = gp.predict(X_test, return_std=True)
prediction_ridge = ridge.predict(X_test)
prediction_lasso = lasso.predict(X_test)


# mse_linear = mean_squared_error(y_test, prediction_linear)
# mse_svm = mean_squared_error(y_test, prediction_svm)
# mse_gaussian = mean_squared_error(y_test, prediction_gaussian)
# mse_ridge = mean_squared_error(y_test, prediction_ridge)
# mse_lasso = mean_squared_error(y_test, prediction_lasso)

# print("\n\nMSE Linear: ", mse_linear)
# print("MSE Support vector regression: ", mse_svm)
# print("MSE Gaussian: ", mse_gaussian)
# print("MSE Ridge: ", mse_ridge)
# print("MSE Lasso: ", mse_lasso)


print("\n\nLinear Prediction: ", prediction_linear)
print("Support vector regression Prediction: ", prediction_svm)
print("Gaussian Prediction: ", prediction_gaussian)
print("Ridge Prediction: ", prediction_ridge)
print("Lasso Prediction: ", prediction_lasso)


def plot_predicted_data(x_test, y_test, predict, model):
    x_column = x_test[:, 0]
    plt.scatter(y_test, predict, color='blue', label='Actual vs. Predicted SPI')
    plt.xlabel('Actual SPI')
    plt.ylabel('Predicted SPI')
    plt.title(f'Actual vs. Predicted SPI {model}')

    plt.scatter(x_column, y_test, color="red", label="ACTUAL DATA")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')

    plt.show()

plot_predicted_data(X_test, y_test, prediction_linear, "Linear Regression")
plot_predicted_data(X_test, y_test, prediction_svm, "Support vector Regression")
plot_predicted_data(X_test, y_test, prediction_gaussian, "Gaussian process Regression")
plot_predicted_data(X_test, y_test, prediction_ridge, "Ridge Regression")
plot_predicted_data(X_test, y_test, prediction_lasso, "Lasso Regression")




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
print("Predicted Output:", predicted_output[0][0])



