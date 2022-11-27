import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv("data/heart.csv")
y = df_train['output']

def knn():
  # Preprocessing Min-Max Scaler
  df_train_pre = joblib.load('model/df_train_pre.sav')
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
  scores = {}
  for i in range(1, 20+1):
      KN = KNeighborsClassifier(n_neighbors = i)
      KN.fit(x_train, y_train)
      y_pred = KN.predict(x_test)
      scores[i] = accuracy_score(y_test, y_pred)
      
  best_k = max(scores, key=scores.get)
  st.success(f"K Terbaik : {best_k} berada di Index : {best_k-1}, Akurasi yang dihasilkan K-NN = {max(scores.values())* 100}%")
  st.write(df_train_pre)
  
  # Create Chart 
  st.write('Dari proses pemodelan yang telah di lakukan menghasilkan grafik sebagai berikut')
  accuration_k = np.array(list(scores.values()))
  chart_data = pd.DataFrame(accuration_k, columns=['Score Akurasi'])
  st.line_chart(chart_data)
  
  # Save Model
  knn = KNeighborsClassifier(n_neighbors=best_k)
  knn.fit(x_train, y_train)
  joblib.dump(knn, 'model/knn_model.sav') # Menyimpan Model ke dalam folder model

def nb():
  # Preprocessing Min-Max Scaler
  df_train_pre = joblib.load('model/df_train_pre.sav')
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
  
  nb = GaussianNB()
  nb.fit(x_train, y_train)
  # Save Model
  joblib.dump(nb, 'model/nb_model.sav') # Menyimpan Model ke dalam folder model
  
  y_pred = nb.predict(x_test)
  akurasi = accuracy_score(y_test,y_pred)
  
  st.success(f'Akurasi yang dihasilkan Naive-Bayes = {akurasi*100}%')
  st.write(df_train_pre)

def dtc():
  # Preprocessing Min-Max Scaler
  df_train_pre = joblib.load('model/df_train_pre.sav')
  x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
  
  dtc = DecisionTreeClassifier()
  dtc.fit(x_train, y_train)
  # Save Model
  joblib.dump(dtc, 'model/dtc_model.sav') # Menyimpan Model ke dalam folder model
  
  y_pred = dtc.predict(x_test)
  akurasi = accuracy_score(y_test,y_pred)
  
  st.success(f'Akurasi yang dihasilkan Decision Tree = {akurasi*100}%')
  st.write(df_train_pre)