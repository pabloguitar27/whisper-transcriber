from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# Cargar el modelo de Whisper
model = whisper.load_model("base")  # Cambiar a "medium" o "large" según lo necesario

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Guardar el archivo subido
    file = request.files['file']
    file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)

    # Transcribir el archivo
    result = model.transcribe(file_path)
    os.remove(file_path)  # Eliminar el archivo después del procesamiento

    # Devolver la transcripción
    return jsonify({"text": result["text"]}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
