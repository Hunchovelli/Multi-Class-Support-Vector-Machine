import librosa
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

audio_dataset_path = "Data\genres_original"
metadata = pd.read_csv("Data\\features_30_sec.csv")

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='Kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)

    return mfccs_scaled_features

extracted_features = []

for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), str(row['label']), str(row['filename']))
    final_class_labels = row['label']
    data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature','class'])

X = np.array(extracted_features_df['feature'].to_list())
y = np.array(extracted_features_df['class'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

rbf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
sigmoid = svm.SVC(kernel='sigmoid', C=1).fit(X_train, y_train)
linear = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sigmoid_pred = sigmoid.predict(X_test)
linear_pred = linear.predict(X_test)

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('\nAccuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('\nAccuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

sigmoid_accuracy = accuracy_score(y_test, sigmoid_pred)
sigmoid_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('\nAccuracy (Sigmoid Kernel): ', "%.2f" % (sigmoid_accuracy*100))
print('F1 (Sigmoid Kernel): ', "%.2f" % (sigmoid_f1*100))

linear_accuracy = accuracy_score(y_test, linear_pred)
linear_f1 = f1_score(y_test, linear_pred, average='weighted')
print('\nAccuracy (Linear Kernel): ', "%.2f" % (linear_accuracy*100))
print('F1 (Linear Kernel): ', "%.2f" % (linear_f1*100))




