import streamlit as st
import joblib
from sklearn.metrics import accuracy_score

output = {0 : 'False', 1 : 'True'}


def knn(data, data_scaler):
  model = joblib.load('model/knn_model.sav')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset tanpa di lakukan normalisasi Min-Max Scaler')
  y_pred = model.predict(data)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {output[y_pred[0]]}')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset dengan di lakukan normalisasi Min-Max Scaler')
  y_pred_scaler = model.predict(data_scaler)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {output[y_pred_scaler[0]]}')

def nb(data, data_scaler):
  model = joblib.load('model/nb_model.sav')

  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset tanpa di lakukan normalisasi Min-Max Scaler')
  y_pred = model.predict(data)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {output[y_pred[0]]}')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset dengan di lakukan normalisasi Min-Max Scaler')
  y_pred_scaler = model.predict(data_scaler)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {output[y_pred_scaler[0]]}')

def dtc(data, data_scaler):
  model = joblib.load('model/dtc_model.sav')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset tanpa di lakukan normalisasi Min-Max Scaler')
  y_pred = model.predict(data)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {output[y_pred[0]]}')
  
  st.write('Hasil Prediksi yang di dapatkan jika menggunakan dataset dengan di lakukan normalisasi Min-Max Scaler')
  y_pred_scaler = model.predict(data_scaler)
  st.success(f'Dengan Spesifikasi Yang telah di inputkan Harga Handphone Termasuk ke dalam kategori : {output[y_pred_scaler[0]]}')
