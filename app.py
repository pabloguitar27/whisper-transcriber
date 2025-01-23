from flask import Flask, request, render_template, send_file
import whisper
import os

app = Flask(__name__)

# Configuración del modelo
model = whisper.load_model("medium")

@app.route('/')
def index():
    # Sirve la página principal con el formulario
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file provided", 400
    
    # Guardar el archivo subido
    file = request.files['file']
    input_file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    file.save(input_file_path)
    
    # Configuración de salida
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.splitext(file.filename)[0] + ".srt")

    # Procesar con Whisper
    result = model.transcribe(
        input_file_path,
        language="en",
        output_format="srt",
        max_line_width=30,
        max_words_per_line=7,
        max_line_count=1,
        no_speech_threshold=0.5,
        logprob_threshold=-1.0
    )

    # Guardar los subtítulos generados
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # Eliminar el archivo de entrada
    os.remove(input_file_path)

    # Devolver un enlace para descargar el archivo
    return f"""
        <h1>Archivo procesado con éxito</h1>
        <a href="/download/{os.path.basename(output_file_path)}" download>Descargar subtítulos</a>
    """

@app.route('/download/<filename>')
def download_file(filename):
    # Ruta completa del archivo
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
