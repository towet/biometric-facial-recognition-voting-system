from flask import Flask, render_template, request, redirect, url_for, session, flash
import cv2
import os
import json
import base64
from roboflow import Roboflow

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

# Load validation data from validation.json
with open('validation.json', 'r') as f:
    validation_data = json.load(f)

# Initialize Roboflow model
rf = Roboflow(api_key="ZvUrmQtRaqgJmxz6IRRh")
project = rf.workspace().project("metric-ngxar")
model = project.version(2).model

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in validation_data and validation_data[username] == password:
            session['username'] = username
            return redirect(url_for('capture'))
        
        flash("Invalid credentials")
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/capture')
def capture():
    if 'username' not in session:
        flash("Please log in first")
        return redirect(url_for('login'))
    
    return render_template('capture.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'username' not in session:
        flash("Please log in first")
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash("No file selected")
        return redirect(url_for('capture'))

    file = request.files['image']
    
    if file.filename == '':
        flash("No file selected")
        return redirect(url_for('capture'))

    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    
    return redirect(url_for('predict', file_path=file_path.replace('\\', '/')))

@app.route('/predict/<path:file_path>')
def predict(file_path):
    if 'username' not in session:
        flash("Please log in first")
        return redirect(url_for('login'))

    prediction = model.predict(file_path, confidence=4.0, overlap=30).json()
    
    image = cv2.imread(file_path)
    
    if image is None:
        flash("Failed to load image")
        return redirect(url_for('capture'))
    
    height, width = image.shape[:2]
    
    for pred in prediction['predictions']:
        x = int(pred['x'])
        y = int(pred['y'])
        
        box_width = int(pred['width'])
        box_height = int(pred['height'])
        
        x1 = max(0, x - box_width // 2)
        y1 = max(0, y - box_height // 2)
        
        x2 = min(width, x + box_width // 2)
        y2 = min(height, y + box_height // 2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    result_file_path = os.path.join('static', 'result_' + os.path.basename(file_path))
    
    cv2.imwrite(result_file_path, image)

    _, buffer = cv2.imencode('.png', image)
    
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    detected_class = prediction['predictions'][0]['class'] if prediction['predictions'] else None
    
    is_verified = detected_class and session['username'].lower() == detected_class.lower()
    
    message = "Verification complete." if is_verified else "Authorization denied."

    return render_template('results.html', 
                           file_path=result_file_path.replace('\\', '/'), 
                           prediction=prediction,
                           img_base64=img_base64,
                           is_verified=is_verified,
                           message=message)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
