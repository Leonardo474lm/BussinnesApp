import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Modelo de Machine Learning para la Predicción de Riesgo de Ataque Cardíaco")

uploaded_file = st.file_uploader("Sube tu archivo CSV 'heart_attack_prediction_dataset.csv'", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,  sep=",")
    st.success("Archivo cargado exitosamente.")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())
else:
    st.info("Por favor, sube un archivo CSV para continuar.")
    st.stop() # Detiene la ejecución si no hay archivo

file_path = "heart_attack_prediction_dataset.csv" # Asegúrate de que el archivo esté en la misma carpeta o especifica la ruta completa
df = pd.read_csv(file_path, sep=",")
st.dataframe(df.head())

st.subheader("Información del Dataset")
st.write(df.info()) # Esto imprimirá la información en la consola de tu terminal, no en la app
# Para mostrarla en la app de Streamlit, podrías redirigir la salida de df.info() a un StringIO
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Dimensiones del Dataset")
st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

st.subheader("Detección de Filas Duplicadas")
if df.duplicated().any():
    st.write("Se encontraron filas duplicadas:")
    st.dataframe(df[df.duplicated()])
else:
    st.write("No se encontraron filas duplicadas.")


