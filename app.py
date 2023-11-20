from flask import Flask, request, render_template, send_from_directory, jsonify
import cv2
import os
import numpy as np

app = Flask(__name__)

def recognize_faces(image_data):
    # Implement your face recognition logic here
    # This function should return a list of recognized names
    
    # Dummy data for demonstration
    recognized_names = ["John Doe", "Alice Smith", "Bob Johnson"]
    
    return recognized_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/recognize', methods=['POST'])
def recognize():
    image_data = request.json['image']
    
    # Save the received image temporarily
    with open('temp.jpg', 'wb') as f:
        f.write(image_data)

    # Load the saved image using OpenCV
    img = cv2.imread('temp.jpg')

    # Call the face recognition function
    recognized_names = recognize_faces(img)
    
    # Generate CSV data from recognized names
    csv_data = "\n".join(recognized_names)
    
    # Save the CSV file
    with open('static/results/attendance.csv', 'w') as csv_file:
        csv_file.write(csv_data)

    # Remove the temporary image
    os.remove('temp.jpg')

    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
