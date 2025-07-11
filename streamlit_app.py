import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.3f}'.format

df = pd.read_csv("/content/drive/MyDrive/TF/Grupo_Bussiness/heart_attack_prediction_dataset.csv", encoding='ISO-8859-1', sep=",")
df.head(4)
df.info()
print(df.shape)
