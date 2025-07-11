import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
pd.options.display.float_format = '{:.3f}'.format
url = st.text_input("Pega el enlace público del archivo CSV en Google Drive")

if url:
    try:
        file_id = url.split("/")[-2]
        dwn_url = f'https://drive.google.com/uc?id={file_id}'
        df = pd.read_csv(dwn_url)
        st.write("Datos cargados:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
