import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
st.set_page_config(layout="wide")
st.title("👨‍⚕️ Predicción de Riesgo de Ataque Cardíaco con Random Forest")
st.markdown("---")
st.sidebar.subheader("Carga de Datos y Entrenamiento del Modelo")
st.sidebar.info("Por favor, sube tu archivo 'heart_attack_prediction_dataset.csv' para entrenar el modelo.")

df = None
rf_model = None
scaler = None
imputer = None
one_hot_encoder = Noneimport pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# --- 0. Configuración inicial de Streamlit ---
st.set_page_config(layout="wide")
st.title("👨‍⚕️ Predicción de Riesgo de Ataque Cardíaco con Random Forest")
st.markdown("---")

# --- 1. Carga de Datos y Entrenamiento del Modelo de Random Forest ---
st.sidebar.subheader("Carga de Datos y Entrenamiento del Modelo")
st.sidebar.info("Por favor, sube tu archivo 'heart_attack_prediction_dataset.csv' para entrenar el modelo.")

df = None
rf_model = None
scaler = None
imputer = None
one_hot_encoder = None
model_trained = False
model_features_order = None

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV 'heart_attack_prediction_dataset.csv'", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', sep=",")
        st.sidebar.success("Dataset cargado exitosamente.")
        st.markdown("### 📊 Análisis Exploratorio de Datos (EDA) - Primeras 5 filas")
        st.dataframe(df.head())

        # --- Limpiar y estandarizar nombres de columnas ---
        df.columns = df.columns.str.strip().str.lower()
        st.sidebar.write(f"Nombres de columnas después de la limpieza inicial: {df.columns.tolist()}")


        # --- Validación y preprocesamiento inicial de la columna 'sex' ---
        if 'sex' not in df.columns:
            st.sidebar.error("Error: La columna 'sex' no se encontró en el archivo CSV después de la normalización. Por favor, asegúrate de que el nombre de la columna sea 'Sex' o 'sex' en tu archivo original.")
            st.stop()

        df['sex'] = df['sex'].astype(str).str.capitalize().replace('Nan', np.nan)
        st.sidebar.write(f"Valores únicos en la columna 'sex' después de la limpieza: {df['sex'].unique()}")

        df_male = df[df['sex'] == 'Male'].copy()

        if df_male.empty:
            st.sidebar.warning("Advertencia: No se encontraron registros con 'sex' igual a 'Male' después del preprocesamiento. El modelo se entrenará con un DataFrame vacío.")
            st.stop()

        # --- Entrenamiento del Modelo de Random Forest ---
        st.sidebar.write("Entrenando el modelo de Random Forest...")

        df_male.rename(columns={"diabates":"diabetes"}, inplace=True)

        cols_to_numeric = ['bmi', 'sedentary hours per day', 'exercise hours per week']
        for col in cols_to_numeric:
            if col in df_male.columns:
                df_male[col] = pd.to_numeric(df_male[col], errors="coerce")
            else:
                st.sidebar.warning(f"La columna numérica '{col}' no se encontró en el dataset.")


        binary_cols_map = {
            'smoking': {'No': 0, 'Yes': 1},
            'alcohol consumption': {'No': 0, 'Yes': 1},
            'family history': {'No': 0, 'Yes': 1},
            'diabetes': {'No': 0, 'Yes': 1},
            'previous heart problems': {'No': 0, 'Yes': 1},
            'medication use': {'No': 0, 'Yes': 1}
        }
        for col, mapping in binary_cols_map.items():
            if col in df_male.columns:
                df_male[col] = df_male[col].fillna('No').map(mapping)
            else:
                st.sidebar.warning(f"La columna binaria '{col}' no se encontró en el dataset.")

        features_list = [
            'age', 'cholesterol', 'heart rate', 'diabetes', 'family history',
            'smoking', 'alcohol consumption', 'exercise hours per week',
            'diet', 'previous heart problems', 'medication use', 'stress level',
            'bmi', 'sedentary hours per day', 'income', 'triglycerides',
            'physical activity days per week', 'sleep hours per day',
            'hemisphere'
        ]
        target = 'heart attack risk'

        all_required_cols = features_list + [target]
        missing_cols = [col for col in all_required_cols if col not in df_male.columns]
        if missing_cols:
            st.sidebar.error(f"Error: Faltan las siguientes columnas en tu CSV para entrenar el modelo: {', '.join(missing_cols)}")
            st.stop()

        df_male_model = df_male[features_list + [target]].copy()

        columnas_onehot = ['diet', 'hemisphere']
        
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', dtype=int)
        
        cols_to_encode_present = [col for col in columnas_onehot if col in df_male_model.columns]
        if cols_to_encode_present:
            df_oneHE_array = one_hot_encoder.fit_transform(df_male_model[cols_to_encode_present])
            df_oneHE = pd.DataFrame(df_oneHE_array, columns=one_hot_encoder.get_feature_names_out(cols_to_encode_present), index=df_male_model.index)
            df_male_model = pd.concat([df_male_model.drop(columns=cols_to_encode_present), df_oneHE], axis=1)
        else:
            st.sidebar.warning("No se encontraron columnas para One-Hot Encoding (diet, hemisphere).")


        X = df_male_model.drop(target, axis=1)
        y = df_male_model[target]

        X = X.dropna(axis=1, how="all")

        X.columns = X.columns.str.strip().str.lower()

        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        st.sidebar.write(f"X_imputed.head() después de imputación:\n{X_imputed.head()}")
        st.sidebar.write(f"X_imputed.dtypes después de imputación:\n{X_imputed.dtypes}")

        model_features_order = X_imputed.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.25, random_state=42, stratify=y
        )

        st.sidebar.write("Aplicando SMOTE para balancear las clases de entrenamiento...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        st.sidebar.write(f"Clases de entrenamiento originales: {y_train.value_counts()}")
        st.sidebar.write(f"Clases de entrenamiento después de SMOTE: {y_train_smote.value_counts()}")


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200)
        rf_model.fit(X_train_scaled, y_train_smote)
        model_trained = True
        st.sidebar.success("Modelo de Random Forest entrenado exitosamente con SMOTE.")

        rf_pred = rf_model.predict(X_test_scaled)
        rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, rf_pred)
        auc = roc_auc_score(y_test, rf_proba)
        try:
            precision = precision_score(y_test, rf_pred)
            recall = recall_score(y_test, rf_pred)
            f1 = f1_score(y_test, rf_pred)
            st.sidebar.markdown("---")
            st.sidebar.subheader("Métricas del Modelo Entrenado (Solo Hombres)")
            st.sidebar.write(f"Accuracy: `{acc:.3f}`")
            st.sidebar.write(f"AUC: `{auc:.3f}`")
            st.sidebar.write(f"Precision: `{precision:.3f}`")
            st.sidebar.write(f"Recall: `{recall:.3f}`")
            st.sidebar.write(f"F1-score: `{f1:.3f}`")
            st.sidebar.write("Reporte de Clasificación:")
            st.sidebar.text(classification_report(y_test, rf_pred))
            st.sidebar.write("Matriz de Confusión:")
            st.sidebar.text(confusion_matrix(y_test, rf_pred))

            # --- Graficar Matriz de Confusión ---
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(y_test, rf_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Riesgo (0)', 'Riesgo (1)'],
                        yticklabels=['No Riesgo (0)', 'Riesgo (1)'])
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.title('Matriz de Confusión')
            st.pyplot(plt)
            plt.clf()

            # --- Graficar Curva ROC ---
            st.subheader("Curva ROC")
            fpr, tpr, thresholds = roc_curve(y_test, rf_proba)
            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title('Curva Característica Operativa del Receptor (ROC)')
            plt.legend(loc="lower right")
            st.pyplot(plt)
            plt.clf()

            # --- Mostrar Importancia de Características del Random Forest ---
            st.subheader("Importancia de las Características del Modelo Random Forest")
            feature_importances_df = pd.DataFrame({
                'Característica': model_features_order,
                'Importancia': rf_model.feature_importances_
            })
            feature_importances_df = feature_importances_df.sort_values(by='Importancia', ascending=False)
            
            st.dataframe(feature_importances_df)
            st.info("Interpretación de la importancia de características:\n"
                    "- Un valor más alto indica que la característica es más relevante para las predicciones del modelo.\n\n"
                    "**Nota sobre resultados contraintuitivos (ej. ejercicio):** Si una característica como 'horas de ejercicio' muestra una correlación inesperada (positiva con el riesgo), esto a menudo indica una relación compleja o un sesgo en los datos. El modelo aprende de los patrones existentes. Podría ser que en tu dataset, las personas con mayor riesgo ya existente (por otras condiciones) sean las que realizan más ejercicio para manejar su salud. Esto no significa que el ejercicio cause riesgo, sino que el riesgo preexistente motiva el ejercicio. Se recomienda una exploración de datos más profunda para entender estas relaciones.")


        except Exception as e:
            st.sidebar.warning(f"No se pudieron calcular o graficar todas las métricas: {e}")

    except Exception as e:
        st.sidebar.error(f"Error al cargar o procesar el archivo CSV para el entrenamiento. Por favor, verifica el archivo y su codificación. Error: {e}")
        model_trained = False
else:
    st.sidebar.info("Por favor, sube un CSV para entrenar el modelo y habilitar la predicción.")

st.markdown("---")
st.header("📝 Ingresa los Datos del Paciente")
st.warning("Este modelo ha sido entrenado **exclusivamente con datos de pacientes masculinos**. Las predicciones para mujeres pueden no ser precisas.")

# --- 3. Widgets para la Entrada de Datos del Usuario ---
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Edad", 18, 100, 50)
    sex_input = st.selectbox("Género", ['Male', 'Female'], index=0)
    if sex_input == 'Female':
        st.warning("El modelo está optimizado para pacientes masculinos. La predicción para mujeres puede no ser precisa.")
    
    cholesterol = st.number_input("Colesterol (mg/dL)", 100, 600, 200)
    heart_rate = st.number_input("Ritmo Cardíaco (ppm)", 40, 200, 70)
    diabetes = st.selectbox("Diabetes", ["No", "Sí"])
    family_history = st.selectbox("Historia Familiar de Enfermedades Cardíacas", ["No", "Sí"])
    smoking = st.selectbox("Fumador", ["No", "Sí"])

with col2:
    alcohol_consumption = st.selectbox("Consumo de Alcohol", ["No", "Sí"])
    exercise_hours_per_week = st.number_input("Horas de Ejercicio por Semana", 0.0, 20.0, 3.0, step=0.5)
    diet = st.selectbox("Dieta", ["Average", "Healthy", "Unhealthy"])
    previous_heart_problems = st.selectbox("Problemas Cardíacos Previos", ["No", "Sí"])
    medication_use = st.selectbox("Uso de Medicación", ["No", "Sí"])
    stress_level = st.slider("Nivel de Estrés (1-10)", 1, 10, 5)

with col3:
    bmi = st.number_input("IMC (Índice de Masa Corporal)", 15.0, 50.0, 25.0, step=0.1)
    sedentary_hours_per_day = st.number_input("Horas Sedentarias por Día", 0.0, 20.0, 8.0, step=0.5)
    income = st.number_input("Ingresos", 0, 1000000, 50000)
    triglycerides = st.number_input("Triglicéridos (mg/dL)", 50, 1000, 150)
    physical_activity_days_per_week = st.slider("Días de Actividad Física por Semana", 0, 7, 3)
    sleep_hours_per_day = st.number_input("Horas de Sueño por Día", 0.0, 12.0, 7.0, step=0.5)
    hemisphere = st.selectbox("Hemisferio", ["Northern", "Southern"])


# --- 4. Botón de Predicción y Lógica ---
st.markdown("---")
if st.button("Calcular Riesgo de Ataque Cardíaco"):
    if not model_trained:
        st.error("El modelo no ha sido entrenado. Por favor, sube el archivo CSV primero.")
    else:
        # 4.1 Preprocesamiento de los datos de entrada del usuario
        user_input_raw = pd.DataFrame([[
            age,
            cholesterol,
            heart_rate,
            1 if diabetes == "Sí" else 0,
            1 if family_history == "Sí" else 0,
            1 if smoking == "Sí" else 0,
            1 if alcohol_consumption == "Sí" else 0,
            exercise_hours_per_week,
            diet,
            1 if previous_heart_problems == "Sí" else 0,
            1 if medication_use == "Sí" else 0,
            stress_level,
            bmi,
            sedentary_hours_per_day,
            income,
            triglycerides,
            physical_activity_days_per_week,
            sleep_hours_per_day,
            hemisphere
        ]], columns=[
            'age', 'cholesterol', 'heart rate', 'diabetes', 'family history',
            'smoking', 'alcohol consumption', 'exercise hours per week',
            'diet', 'previous heart problems', 'medication use', 'stress level',
            'bmi', 'sedentary hours per day', 'income', 'triglycerides',
            'physical activity days per week', 'sleep hours per day',
            'hemisphere'
        ])
        
        user_categorical_data = user_input_raw[['diet', 'hemisphere']]
        user_oneHE_array = one_hot_encoder.transform(user_categorical_data)
        user_oneHE = pd.DataFrame(user_oneHE_array, columns=one_hot_encoder.get_feature_names_out(['diet', 'hemisphere']), index=user_input_raw.index)

        user_numerical_data = user_input_raw.drop(columns=['diet', 'hemisphere'])
        user_processed_data = pd.concat([user_numerical_data, user_oneHE], axis=1)

        temp_imputer_input = pd.DataFrame(0, index=user_processed_data.index, columns=imputer.feature_names_in_)
        for col in temp_imputer_input.columns:
            if col in user_processed_data.columns:
                temp_imputer_input[col] = user_processed_data[col]
        
        user_imputed_data_array = imputer.transform(temp_imputer_input)
        user_imputed_data = pd.DataFrame(user_imputed_data_array, columns=imputer.feature_names_in_, index=user_input_raw.index)

        final_user_data = pd.DataFrame(0, index=user_input_raw.index, columns=model_features_order)
        for col in final_user_data.columns:
            if col in user_imputed_data.columns:
                final_user_data[col] = user_imputed_data[col]

        user_data_scaled = scaler.transform(final_user_data)

        try:
            prediction = rf_model.predict(user_data_scaled)
            prediction_proba = rf_model.predict_proba(user_data_scaled)

            st.subheader("🎉 Resultado de la Predicción")
            if prediction[0] == 1:
                st.error(f"¡Atención! Se predice un **ALTO RIESGO** de ataque cardíaco para este paciente.")
                st.write(f"Probabilidad de Riesgo: **{prediction_proba[0][1]*100:.2f}%**")
                st.info("Se recomienda encarecidamente consultar a un profesional médico.")
            else:
                st.success(f"Se predice **BAJO RIESGO** de ataque cardíaco para este paciente.")
                st.write(f"Probabilidad de No Riesgo: **{prediction_proba[0][0]*100:.2f}%**")
                st.info("Mantener un estilo de vida saludable es clave para la prevención.")

            st.markdown("---")
            st.write("Valores ingresados por el usuario (después del preprocesamiento para el modelo):")
            st.dataframe(final_user_data)

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción. Asegúrate de que los datos de entrada coincidan con lo que espera el modelo. Error: {e}")

# --- Pie de página (Opcional) ---
st.markdown("---")
st.caption("Esta aplicación es solo con fines demostrativos y no debe reemplazar el consejo médico profesional.")
model_trained = False
model_features_order = None

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV 'heart_attack_prediction_dataset.csv'", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', sep=",")
        st.sidebar.success("Dataset cargado exitosamente.")
        st.markdown("### 📊 Análisis Exploratorio de Datos (EDA) - Primeras 5 filas")
        st.dataframe(df.head())
        df.columns = df.columns.str.strip().str.lower()

        if 'sex' not in df.columns:
            st.sidebar.error("Error: La columna 'sex' no se encontró en el archivo CSV después de la normalización. Por favor, asegúrate de que el nombre de la columna sea 'Sex' o 'sex' en tu archivo original.")
            st.stop()

        df['sex'] = df['sex'].astype(str).str.capitalize().replace('Nan', np.nan)

        df_male = df[df['sex'] == 'Male'].copy()

        if df_male.empty:
            st.sidebar.warning("Advertencia: No se encontraron registros con 'sex' igual a 'Male' después del preprocesamiento. El modelo se entrenará con un DataFrame vacío.")
            st.stop()

        # --- Entrenamiento del Modelo de Random Forest ---
        st.sidebar.write("Entrenando el modelo de Random Forest...")

        df_male.rename(columns={"diabates":"diabetes"}, inplace=True)

        cols_to_numeric = ['bmi', 'sedentary hours per day', 'exercise hours per week']
        for col in cols_to_numeric:
            if col in df_male.columns:
                df_male[col] = pd.to_numeric(df_male[col], errors="coerce")
            else:
                st.sidebar.warning(f"La columna numérica '{col}' no se encontró en el dataset.")


        binary_cols_map = {
            'smoking': {'No': 0, 'Yes': 1},
            'alcohol consumption': {'No': 0, 'Yes': 1},
            'family history': {'No': 0, 'Yes': 1},
            'diabetes': {'No': 0, 'Yes': 1},
            'previous heart problems': {'No': 0, 'Yes': 1},
            'medication use': {'No': 0, 'Yes': 1}
        }
        for col, mapping in binary_cols_map.items():
            if col in df_male.columns:
                df_male[col] = df_male[col].fillna('No').map(mapping)
            else:
                st.sidebar.warning(f"La columna binaria '{col}' no se encontró en el dataset.")

        features_list = [
            'age', 'cholesterol', 'heart rate', 'diabetes', 'family history',
            'smoking', 'alcohol consumption', 'exercise hours per week',
            'diet', 'previous heart problems', 'medication use', 'stress level',
            'bmi', 'sedentary hours per day', 'income', 'triglycerides',
            'physical activity days per week', 'sleep hours per day',
            'hemisphere'
        ]
        target = 'heart attack risk'

        all_required_cols = features_list + [target]
        missing_cols = [col for col in all_required_cols if col not in df_male.columns]
        if missing_cols:
            st.sidebar.error(f"Error: Faltan las siguientes columnas en tu CSV para entrenar el modelo: {', '.join(missing_cols)}")
            st.stop()

        df_male_model = df_male[features_list + [target]].copy()

        columnas_onehot = ['diet', 'hemisphere']
        
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', dtype=int)
        
        cols_to_encode_present = [col for col in columnas_onehot if col in df_male_model.columns]
        if cols_to_encode_present:
            df_oneHE_array = one_hot_encoder.fit_transform(df_male_model[cols_to_encode_present])
            df_oneHE = pd.DataFrame(df_oneHE_array, columns=one_hot_encoder.get_feature_names_out(cols_to_encode_present), index=df_male_model.index)
            df_male_model = pd.concat([df_male_model.drop(columns=cols_to_encode_present), df_oneHE], axis=1)
        else:
            st.sidebar.warning("No se encontraron columnas para One-Hot Encoding (diet, hemisphere).")


        X = df_male_model.drop(target, axis=1)
        y = df_male_model[target]

        X = X.dropna(axis=1, how="all")

        X.columns = X.columns.str.strip().str.lower()

        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        

        model_features_order = X_imputed.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.25, random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200)
        rf_model.fit(X_train_scaled, y_train_smote)
        model_trained = True
        st.sidebar.success("Modelo de Random Forest entrenado exitosamente con SMOTE.")

        rf_pred = rf_model.predict(X_test_scaled)
        rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, rf_pred)
        auc = roc_auc_score(y_test, rf_proba)
        try:
            precision = precision_score(y_test, rf_pred)
            recall = recall_score(y_test, rf_pred)
            f1 = f1_score(y_test, rf_pred)
            st.sidebar.markdown("---")
            st.sidebar.subheader("Métricas del Modelo Entrenado (Solo Hombres)")
            st.sidebar.write(f"Accuracy: `{acc:.3f}`")
            st.sidebar.write(f"AUC: `{auc:.3f}`")
            st.sidebar.write(f"Precision: `{precision:.3f}`")
            st.sidebar.write(f"Recall: `{recall:.3f}`")
            st.sidebar.write(f"F1-score: `{f1:.3f}`")
            st.sidebar.write("Reporte de Clasificación:")
            st.sidebar.text(classification_report(y_test, rf_pred))
            st.sidebar.write("Matriz de Confusión:")
            st.sidebar.text(confusion_matrix(y_test, rf_pred))


            # --- Mostrar Importancia de Características del Random Forest ---
            st.subheader("Importancia de las Características del Modelo Random Forest")
            feature_importances_df = pd.DataFrame({
                'Característica': model_features_order,
                'Importancia': rf_model.feature_importances_
            })
            feature_importances_df = feature_importances_df.sort_values(by='Importancia', ascending=False)
            
            st.dataframe(feature_importances_df)
            st.info("Interpretación de la importancia de características:\n"
                    "- Un valor más alto indica que la característica es más relevante para las predicciones del modelo.\n\n"
                    "**Nota sobre resultados contraintuitivos (ej. ejercicio):** Si una característica como 'horas de ejercicio' muestra una correlación inesperada (positiva con el riesgo), esto a menudo indica una relación compleja o un sesgo en los datos. El modelo aprende de los patrones existentes. Podría ser que en tu dataset, las personas con mayor riesgo ya existente (por otras condiciones) sean las que realizan más ejercicio para manejar su salud. Esto no significa que el ejercicio cause riesgo, sino que el riesgo preexistente motiva el ejercicio. Se recomienda una exploración de datos más profunda para entender estas relaciones.")


        except Exception as e:
            st.sidebar.warning(f"No se pudieron calcular o graficar todas las métricas: {e}")

    except Exception as e:
        st.sidebar.error(f"Error al cargar o procesar el archivo CSV para el entrenamiento. Por favor, verifica el archivo y su codificación. Error: {e}")
        model_trained = False
else:
    st.sidebar.info("Por favor, sube un CSV para entrenar el modelo y habilitar la predicción.")

st.markdown("---")
st.header("📝 Ingresa los Datos del Paciente")
st.warning("Este modelo ha sido entrenado **exclusivamente con datos de pacientes masculinos**. Las predicciones para mujeres pueden no ser precisas.")

# --- 3. Widgets para la Entrada de Datos del Usuario ---
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Edad", 18, 100, 50)
    sex_input = st.selectbox("Género", ['Male', 'Female'], index=0)
    if sex_input == 'Female':
        st.warning("El modelo está optimizado para pacientes masculinos. La predicción para mujeres puede no ser precisa.")
    
    cholesterol = st.number_input("Colesterol (mg/dL)", 100, 600, 200)
    heart_rate = st.number_input("Ritmo Cardíaco (ppm)", 40, 200, 70)
    diabetes = st.selectbox("Diabetes", ["No", "Sí"])
    family_history = st.selectbox("Historia Familiar de Enfermedades Cardíacas", ["No", "Sí"])
    smoking = st.selectbox("Fumador", ["No", "Sí"])

with col2:
    alcohol_consumption = st.selectbox("Consumo de Alcohol", ["No", "Sí"])
    exercise_hours_per_week = st.number_input("Horas de Ejercicio por Semana", 0.0, 20.0, 3.0, step=0.5)
    diet = st.selectbox("Dieta", ["Average", "Healthy", "Unhealthy"])
    previous_heart_problems = st.selectbox("Problemas Cardíacos Previos", ["No", "Sí"])
    medication_use = st.selectbox("Uso de Medicación", ["No", "Sí"])
    stress_level = st.slider("Nivel de Estrés (1-10)", 1, 10, 5)

with col3:
    bmi = st.number_input("IMC (Índice de Masa Corporal)", 15.0, 50.0, 25.0, step=0.1)
    sedentary_hours_per_day = st.number_input("Horas Sedentarias por Día", 0.0, 20.0, 8.0, step=0.5)
    income = st.number_input("Ingresos", 0, 1000000, 50000)
    triglycerides = st.number_input("Triglicéridos (mg/dL)", 50, 1000, 150)
    physical_activity_days_per_week = st.slider("Días de Actividad Física por Semana", 0, 7, 3)
    sleep_hours_per_day = st.number_input("Horas de Sueño por Día", 0.0, 12.0, 7.0, step=0.5)
    hemisphere = st.selectbox("Hemisferio", ["Northern", "Southern"])


# --- 4. Botón de Predicción y Lógica ---
st.markdown("---")
if st.button("Calcular Riesgo de Ataque Cardíaco"):
    if not model_trained:
        st.error("El modelo no ha sido entrenado. Por favor, sube el archivo CSV primero.")
    else:
        # 4.1 Preprocesamiento de los datos de entrada del usuario
        user_input_raw = pd.DataFrame([[
            age,
            cholesterol,
            heart_rate,
            1 if diabetes == "Sí" else 0,
            1 if family_history == "Sí" else 0,
            1 if smoking == "Sí" else 0,
            1 if alcohol_consumption == "Sí" else 0,
            exercise_hours_per_week,
            diet,
            1 if previous_heart_problems == "Sí" else 0,
            1 if medication_use == "Sí" else 0,
            stress_level,
            bmi,
            sedentary_hours_per_day,
            income,
            triglycerides,
            physical_activity_days_per_week,
            sleep_hours_per_day,
            hemisphere
        ]], columns=[
            'age', 'cholesterol', 'heart rate', 'diabetes', 'family history',
            'smoking', 'alcohol consumption', 'exercise hours per week',
            'diet', 'previous heart problems', 'medication use', 'stress level',
            'bmi', 'sedentary hours per day', 'income', 'triglycerides',
            'physical activity days per week', 'sleep hours per day',
            'hemisphere'
        ])
        
        user_categorical_data = user_input_raw[['diet', 'hemisphere']]
        user_oneHE_array = one_hot_encoder.transform(user_categorical_data)
        user_oneHE = pd.DataFrame(user_oneHE_array, columns=one_hot_encoder.get_feature_names_out(['diet', 'hemisphere']), index=user_input_raw.index)

        user_numerical_data = user_input_raw.drop(columns=['diet', 'hemisphere'])
        user_processed_data = pd.concat([user_numerical_data, user_oneHE], axis=1)

        temp_imputer_input = pd.DataFrame(0, index=user_processed_data.index, columns=imputer.feature_names_in_)
        for col in temp_imputer_input.columns:
            if col in user_processed_data.columns:
                temp_imputer_input[col] = user_processed_data[col]
        
        user_imputed_data_array = imputer.transform(temp_imputer_input)
        user_imputed_data = pd.DataFrame(user_imputed_data_array, columns=imputer.feature_names_in_, index=user_input_raw.index)

        final_user_data = pd.DataFrame(0, index=user_input_raw.index, columns=model_features_order)
        for col in final_user_data.columns:
            if col in user_imputed_data.columns:
                final_user_data[col] = user_imputed_data[col]

        user_data_scaled = scaler.transform(final_user_data)

        try:
            prediction = rf_model.predict(user_data_scaled)
            prediction_proba = rf_model.predict_proba(user_data_scaled)

            st.subheader("🎉 Resultado de la Predicción")
            if prediction[0] == 1:
                st.error(f"¡Atención! Se predice un **ALTO RIESGO** de ataque cardíaco para este paciente.")
                st.write(f"Probabilidad de Riesgo: **{prediction_proba[0][1]*100:.2f}%**")
                st.info("Se recomienda encarecidamente consultar a un profesional médico.")
            else:
                st.success(f"Se predice **BAJO RIESGO** de ataque cardíaco para este paciente.")
                st.write(f"Probabilidad de No Riesgo: **{prediction_proba[0][0]*100:.2f}%**")
                st.info("Mantener un estilo de vida saludable es clave para la prevención.")

            st.markdown("---")
            st.write("Valores ingresados por el usuario (después del preprocesamiento para el modelo):")
            st.dataframe(final_user_data)

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción. Asegúrate de que los datos de entrada coincidan con lo que espera el modelo. Error: {e}")
st.markdown("---")
st.caption("Esta aplicación es solo con fines demostrativos y no debe reemplazar el consejo médico profesional.")