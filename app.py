import base64, io, os, cv2, time
import numpy as np
from PIL import Image
# for our model
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
# to retrieve and send back data
from flask import Flask, request, render_template, redirect

# create a variable named app
app = Flask(__name__, static_folder='static')
IMG_SHAPE = (160, 160)

# Configure upload folder (optional)
UPLOAD_FOLDER = 'image_capture'  # Define a folder to save captured images (optional)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        # Log the error and return an appropriate error message
        app.logger.error(f"Error rendering template: {e}")
        return "An error occurred while rendering the template.", 500
    
@app.route('/upload')
def detect():
    try:
        return render_template('upload.html')
    except Exception as e:
        # Log the error and return an appropriate error message
        app.logger.error(f"Error rendering template: {e}")
        return "An error occurred while rendering the template.", 500

@app.route('/capture', methods=['POST'])
def capture_image():
    if request.method == 'POST':
        # Get image data from request (as bytes)
        image_data = request.get_data()
        # Generate unique filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"image_{timestamp}.jpg"

        # Split the data to remove the prefix (optional, depending on client-side implementation)
        # image_data = image_data.split(',')[1]  # If the prefix is always present

        # Decode image data (assuming base64 encoding)
        try:
            decoded_data = base64.b64decode(image_data)
        except Exception as e:
            print("Error decoding image:", e)
            return "Error decoding image data"

        # Save the image
        try:
            with open(f"{app.config['UPLOAD_FOLDER']}/{filename}", "wb") as f:
                f.write(decoded_data)
            return "Image captured successfully!"
        except Exception as e:
            print("Error saving image:", e)
            return "Error saving captured image"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded image file
        uploaded_file = request.files['image']
        # Check if a file is uploaded
        if uploaded_file.filename != '':
            # Get uploaded image data
            img_bytes = uploaded_file.read()
            # Convert image data to a NumPy array
            img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Resize the image (adjust dimensions as needed for your model)
            resized_img = cv2.resize(img, IMG_SHAPE)

            # Normalize the image (common practice for image classification models)
            normalized_img = resized_img / 255.0  # Normalize pixel values between 0 and 1

            # Preprocess further based on your model's requirements (e.g., convert to grayscale)
            # ... (add your specific preprocessing logic here)

            # Make prediction using your model
            prediction = make_prediction(normalized_img)

            # Return prediction results
            return render_template('view/prediction.html', prediction=prediction)

        else:
            # Handle case where no file is uploaded
            return render_template('view/index.html', message="No file selected")
    else:
    # Handle non-POST requests (optional)
        return "This route only accepts POST requests"

def make_prediction(image):
    # Load your pre-trained model
    model_path = "model/model_EfficientNet.h5"  # Replace with your actual path
    loaded_config = tf.keras.models.load_model(model_path).to_config()

    # Fix the axis configuration for BatchNormalization (if needed)
    for layer_index, layer_config in enumerate(loaded_config['layers']):
        if layer_config['class_name'] == 'BatchNormalization' and isinstance(layer_config['config']['axis'], list):
            loaded_config['layers'][layer_index]['config']['axis'] = loaded_config['layers'][layer_index]['config']['axis'][0]

    # Create the model from the modified configuration
    model = tf.keras.models.model_from_config(loaded_config)
    print("[+] Model loaded")

    # Assuming image is already preprocessed (e.g., normalized)

    # Make prediction using the model
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)

    # You can also get the probability of the predicted class
    predicted_proba = prediction[0][predicted_class]

    # Return class label and optionally the probability
    return predicted_class, predicted_proba

if __name__ == '__main__':
    app.run(debug=True)