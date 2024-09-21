from flask import Flask, render_template, request, redirect, url_for, session, flash
import cv2
import os
import json
import base64
from roboflow import Roboflow

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

# Load validation data from validation.json
with open('validation.json', 'r') as f:
    validation_data = json.load(f)

# Initialize Roboflow model
rf = Roboflow(api_key="ZvUrmQtRaqgJmxz6IRRh")
project = rf.workspace().project("metric-ngxar")
model = project.version(2).model

@app.route('/')
def index():
    return render_template('main.html')  # Render main.html as the first page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in validation_data and validation_data[username] == password:
            session['username'] = username
            return redirect(url_for('capture'))
        else:
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

@app.route('/webcam')
def webcam():
    if 'username' not in session:
        flash("Please log in first")
        return redirect(url_for('login'))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        file_path = os.path.join('static', 'captured_image.jpg')
        cv2.imwrite(file_path, frame)
        cap.release()
        return redirect(url_for('predict', file_path=file_path.replace('\\', '/')))
    else:
        cap.release()
        flash("Failed to capture image")
        return redirect(url_for('capture'))

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
        
        x1 = max(0, x - box_width//2)
        y1 = max(0, y - box_height//2)
        x2 = min(width, x + box_width//2)
        y2 = min(height, y + box_height//2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{pred['class']}: {pred['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    result_file_path = os.path.join('static', 'result_' + os.path.basename(file_path))
    cv2.imwrite(result_file_path, image)
    
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Check if the username matches the detected class
    if prediction['predictions']:
        detected_class = prediction['predictions'][0]['class']
        session['last_detected_class'] = detected_class  # Store the detected class in session
        
        # Check if username matches detected class
        if session['username'].lower() == detected_class.lower():
            is_verified = True
            message = "Verification complete. You can cast your vote."
        else:
            is_verified = False
            message = "Authorisation denied you are not authorised to vote attempting to use someone else credential is a crime this voting session has been flagged as suspicious."
    else:
        is_verified = False
        session['last_detected_class'] = None  # No class detected
        message = "No relevant object detected in the image."

    return render_template('results.html', 
                           file_path=result_file_path.replace('\\', '/'), 
                           prediction=prediction,
                           img_base64=img_base64,
                           is_verified=is_verified,
                           message=message)

@app.route('/home')
def home():
    if 'username' not in session:
        flash("Please log in first")
        return redirect(url_for('login'))
    
    if 'last_detected_class' not in session or session['last_detected_class'] is None:
        flash("Please complete the verification process first")
        return redirect(url_for('capture'))
    
    if session['username'].lower() != session['last_detected_class'].lower():
        flash("You are not authorized to access this page. The detected class does not match your username.")
        return redirect(url_for('capture'))
    
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('last_detected_class', None)
    flash("Logged out successfully")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
