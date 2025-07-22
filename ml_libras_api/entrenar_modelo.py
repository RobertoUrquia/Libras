import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Cargar los datos
datos = pd.read_csv('libras.csv')

# Separar variables
X = datos[['Libras']]
y = datos['Kilogramos']

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(modelo, 'modelo_libras_a_kg.pkl')
print("Modelo guardado exitosamente.")
