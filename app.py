import base64, io, os, time
import numpy as np
from PIL import Image

# for our model
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


# to retrieve and send back data
from flask import Flask, request, render_template, redirect

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

def preprocess_image(image, model):
    # Convert RGB image to grayscale
    # image = tf.image.rgb_to_grayscale(image)
    # Resize the image to match model input size
    img = tf.image.resize(image, (model.input_shape[1], model.input_shape[2]))
    # Normalize pixel values (usually between 0 and 1)
    img = img / 255.0
    # Add an extra dimension for batch prediction (if needed)
    img = tf.expand_dims(img, axis=0)
    return img

model_MobileNetV1_v1 = tf.keras.models.load_model('model\\model_MobileNetV1-82_v1.h5')
model_MobileNetV2_v1 = tf.keras.models.load_model('model\\model_MobileNetV2-82_v1.h5')
model_MobileNetV3Small_v1 = tf.keras.models.load_model('model\\model_MobileNetV3Small-82_v1.h5')
model_MobileNetV3Large_v1 = tf.keras.models.load_model('model\\model_MobileNetV3Large-82_v1.h5')
model_VGG16 = tf.keras.models.load_model('model\\model_VGG16-82_v1.h5')   
model_VGG19 = tf.keras.models.load_model('model\\model_VGG19-82_v1.h5')   

    
@app.route('/')
def detect():
    try:
        return render_template('home.html')
    except Exception as e:
        # Log the error and return an appropriate error message
        app.logger.error(f"Error rendering template: {e}")
        return "An error occurred while rendering the template.", 500
    if request.method == 'POST':
        # Get image data from request (as bytes)
        image_data = request.get_data()
        # Generate unique filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"image_{timestamp}.jpg"

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
def predict_image():
    if request.method == 'POST':
        # Get uploaded image file
        uploaded_file = request.files['image']
        
        # Check if a file is uploaded
        if uploaded_file.filename != '':
            # Get uploaded image data
            img_bytes = request.files.get('image')
            # Preprocess the image data in memory
            img_bytes = tf.image.decode_image(img_bytes.read(), channels=3)

            preprocessed_imagev1        = preprocess_image(image=img_bytes, model=model_MobileNetV1_v1)
            preprocessed_imagev2        = preprocess_image(image=img_bytes, model=model_MobileNetV2_v1)
            preprocessed_imagev3small   = preprocess_image(image=img_bytes, model=model_MobileNetV3Small_v1)
            preprocessed_imagev3large   = preprocess_image(image=img_bytes, model=model_MobileNetV3Large_v1)
            preprocessed_imagevgg16   = preprocess_image(image=img_bytes, model=model_VGG16)
            preprocessed_imagevgg19   = preprocess_image(image=img_bytes, model=model_VGG19)

            # Make prediction using the model
            predictions_mobilenetv1         = model_MobileNetV1_v1.predict(preprocessed_imagev1)[0]
            predictions_mobilenetv2         = model_MobileNetV2_v1.predict(preprocessed_imagev2)[0]
            predictions_mobilenetv3small    = model_MobileNetV3Small_v1.predict(preprocessed_imagev3small)[0]
            predictions_mobilenetv3large    = model_MobileNetV3Large_v1.predict(preprocessed_imagev3large)[0]
            predictions_vgg16               = model_VGG16.predict(preprocessed_imagevgg16)[0]
            predictions_vgg19               = model_VGG19.predict(preprocessed_imagevgg19)[0]

            index_m1        = predictions_mobilenetv1.argmax() 
            index_m2        = predictions_mobilenetv2.argmax()
            index_m3small   = predictions_mobilenetv3small.argmax()
            index_m3large   = predictions_mobilenetv3large.argmax()
            index_vgg16     = predictions_vgg16.argmax()
            index_vgg19     = predictions_vgg19.argmax()
            
            predictions = [
                {'nama': class_labels[index_m1],
                'probability': predictions_mobilenetv1[index_m1],
                'nama model': 'MobileNetV1'},
                {'nama': class_labels[index_m2],
                'probability': predictions_mobilenetv2[index_m2],
                'nama model': 'MobileNetV2'},
                {'nama': class_labels[index_m3small],
                'probability': predictions_mobilenetv3small[index_m3small],
                'nama model': 'MobileNetV3Small'},
                {'nama': class_labels[index_m3large],
                'probability': predictions_mobilenetv3large[index_m3large],
                'nama model': 'MobileNetV3Large'},    
                {'nama': class_labels[index_vgg16],
                'probability': predictions_vgg16[index_vgg16],
                'nama model': 'VGG16'},    
                {'nama': class_labels[index_vgg19],
                'probability': predictions_vgg19[index_vgg19],
                'nama model': 'VGG19'},     
            ]
            

            # Return prediction results
            return render_template('prediction.html', prediction=predictions)

        else:
            # Handle case where no file is uploaded
            return render_template('view/index.html', message="No file selected")
    else:
    # Handle non-POST requests (optional)
        return "This route only accepts POST requests"

if __name__ == '__main__':
    app.run(debug=True)