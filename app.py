from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
from pathlib import Path
import subprocess
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'uploads/output'
RESULT_FOLDER = 'result'
YOLO_WEIGHTS = 'D:/fakecurrencywebsite/yolov9/runs/train/exp10/weights/best.pt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

def clean_filename(filename):
    return filename.replace(" ", "_")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = clean_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('run_detection', filename=filename))

    return redirect(url_for('index'))

@app.route('/run_detection/<filename>')
def run_detection(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    command = f'python detect.py --weights {YOLO_WEIGHTS} --source {file_path} --device cpu'
    subprocess.run(command, shell=True)

    return redirect(url_for('loading', filename=filename))

@app.route('/loading/<filename>')
def loading(filename):
    return render_template('loading.html', filename=filename)

@app.route('/copy_image/<filename>')
def copy_image(filename):
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    if os.path.exists(output_image_path):
        shutil.copy(output_image_path, result_image_path)
        return redirect(url_for('view_result', filename=filename))
    else:
        return "Error: Detected image not found in the output folder"

@app.route('/view_result/<filename>')
def view_result(filename):
    output_image_url = url_for('result_file', filename=filename)
    return render_template('result.html', output_image=output_image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
