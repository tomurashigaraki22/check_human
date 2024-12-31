from flask import Flask, request, jsonify
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Initialize OpenCV's pre-trained HOG descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)

# Helper function to preprocess the image for human detection
def preprocess_image(image_bytes):
    # Open image using PIL and convert to a numpy array
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)
    
    # Resize image for faster detection (optional)

    # Convert to grayscale for better performance with HOG
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray, image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_real_human(image_bytes):
    gray, image = preprocess_image(image_bytes)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        print(f"Found {len(faces)} face(s) in the image.")
        return True
    else:
        print("No faces detected in the image.")
        return False




@app.route('/check-image', methods=['POST'])
def check_image():
    # Check if an image file is part of the request
    if 'image' not in request.files:
        print(f"Request files: {request.files}")
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Debugging: log the name of the image received
    print(f"Received image: {image_file.filename}")
    
    # Read the image bytes
    image_bytes = image_file.read()

    # Check if the image contains a real human
    if is_real_human(image_bytes):
        return jsonify({'message': 'This is a real human image!', 'status': 200}), 200
    else:
        return jsonify({'message': 'This is not a real human image.', 'status': 400}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=True)
    
