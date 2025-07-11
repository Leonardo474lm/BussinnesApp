import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#Automcompletar rápido
%config IPCompleter.greedy=True
#Desactivar la notación científica
#Desactivar la notación científica
pd.options.display.float_format = '{:.3f}'.format

#Google Drive
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/TF/Grupo_Bussiness/heart_attack_prediction_dataset.csv", encoding='ISO-8859-1', sep=",")
df.head(4)
df.info()
print(df.shape)
