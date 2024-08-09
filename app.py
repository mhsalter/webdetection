import base64, io, os, time
import numpy as np
from PIL import Image
from io import BytesIO

# for our model
import tensorflow as tf
import cv2

# to retrieve and send back data
from flask import Flask, request, render_template

# create a variable named app
app = Flask(__name__, static_folder='static')
IMG_SHAPE = (224, 224)

class_labels = ['Dadar Gulung',
                'Grontol',
                'Klepon',
                'Kue Lapis',
                'Kue Lumpur',
                'Lumpia',
                'Putu Ayu',
                'Serabi',
                'Wajik']

# Load TFLite models and allocate tensors
interpreter_m1 = tf.lite.Interpreter(model_path="model\\model_MobileNetV1-82_v1.tflite")
interpreter_m2 = tf.lite.Interpreter(model_path="model\\model_MobileNetV2-82_v1.tflite")
interpreter_m3small = tf.lite.Interpreter(model_path="model\\model_MobileNetV3Small-82_v1.tflite")
interpreter_m3large = tf.lite.Interpreter(model_path="model\\model_MobileNetV3Large-82_v1.tflite")
interpreter_vgg16 = tf.lite.Interpreter(model_path="model\\model_VGG16-82_v1.tflite")
interpreter_vgg19 = tf.lite.Interpreter(model_path="model\\model_VGG19-82_v1.tflite")

interpreters = {
    "MobileNetV1": interpreter_m1,
    "MobileNetV2": interpreter_m2,
    "MobileNetV3Small": interpreter_m3small,
    "MobileNetV3Large": interpreter_m3large,
    "VGG16": interpreter_vgg16,
    "VGG19": interpreter_vgg19,
}

for interpreter in interpreters.values():
    interpreter.allocate_tensors()

# Get input details from any interpreter since they share the same input shape
input_details = interpreter_m1.get_input_details()
image_size_x = input_details[0]['shape'][2]
image_size_y = input_details[0]['shape'][1]

IMAGE_MEAN = 0.0
IMAGE_STD = 1.0

def preprocess_image(image: Image.Image):
    # Convert image to numpy array
    image = np.array(image)
    
    # Crop the image to a square
    crop_size = min(image.shape[0], image.shape[1])
    top = (image.shape[0] - crop_size) // 2
    left = (image.shape[1] - crop_size) // 2
    cropped_image = image[top:top + crop_size, left:left + crop_size]

    # Resize the image to the required dimensions with nearest neighbor interpolation
    resized_image = cv2.resize(cropped_image, (image_size_x, image_size_y), interpolation=cv2.INTER_NEAREST)
    # resized_image = cv2.resize(cropped_image, (image_size_x, image_size_y), interpolation=cv2.INTER_LINEAR)
    # resized_image = image.resize((image_size_x, image_size_y), Image.BILINEAR)
    
    # Normalize the image
    normalized_image = (resized_image - IMAGE_MEAN) / IMAGE_STD

    return normalized_image.astype(np.float32)

def predict_with_model(interpreter, image):
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.squeeze(output_data) 

@app.route('/')
def detect():
    try:
        return render_template('home.html')
    except Exception as e:
        # Log the error and return an appropriate error message
        app.logger.error(f"Error rendering template: {e}")
        return "An error occurred while rendering the template.", 500

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Get uploaded image file
        uploaded_file = request.files['image']
        
        # Check if a file is uploaded
        if uploaded_file.filename != '':
            # Save the file temporarily to serve it back as a preview
            preview_image_url = uploaded_file
            
            # Get uploaded image data
            img = Image.open(uploaded_file.stream).convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            preview_image_url = f"data:image/png;base64,{img_base64}"
            
            # Preprocess the image data
            processed_image = preprocess_image(img)
        
            predictions = []
            for model_name, interpreter in interpreters.items():
                prediction = predict_with_model(interpreter, processed_image)
                index = np.argmax(prediction)
                sorted_classes = sorted(zip(class_labels, prediction), key=lambda x: x[1], reverse=True)
                predictions.append({
                    'nama': class_labels[index],
                    'nama model': model_name,
                    'Persentase': round(prediction[index] * 100, 2),
                    'sorted_classes': sorted_classes
                })

            # Return prediction results
            return render_template('home.html', prediction=predictions, preview_image_url=preview_image_url, class_labels=class_labels)

        else:
            # Handle case where no file is uploaded
            return render_template('home.html', message="No file selected")
    else:
        # Handle non-POST requests (optional)
        return "This route only accepts POST requests"

@app.route('/clear_image', methods=['POST'])
def clear_image():
    # Clear the uploaded image file and reset the preview
    try:
        # Retrieve the image file name from the last prediction (or any method you prefer)
        file_name = request.form.get('image_name')
        if file_name:
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        return '', 204
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)