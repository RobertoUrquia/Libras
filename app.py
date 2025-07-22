from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load('modelo_libras_a_kg.pkl')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()

    if 'libras' not in datos:
        return jsonify({'error': 'Debe incluir "libras" en el JSON'}), 400

    libras = np.array(datos['libras']).reshape(-1, 1)
    predicciones = modelo.predict(libras)
    return jsonify({'kilogramos': predicciones.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
