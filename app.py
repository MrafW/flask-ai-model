import base64
import io
import os
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from PIL import Image
from google.cloud import storage
import uuid 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ="./da-kalbe-63ee33c9cdbb.json"

# Configure Google Cloud Storage client (replace placeholders)
BUCKET_NAME = 'da-kalbe-ml-result-png' 
MODEL_FILE = 'densenet.hdf5' # Path to model within the bucket
PRETRAINED_MODEL_FILE = 'pretrained_model.h5'
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# Load the model from Cloud Storage
def load_model_from_gcs():
    blob = bucket.blob(MODEL_FILE)
    blob.download_to_filename(MODEL_FILE) 
    model = load_model(MODEL_FILE)
    
    blob = bucket.blob(PRETRAINED_MODEL_FILE)
    blob.download_to_filename(PRETRAINED_MODEL_FILE)
    pretrained_model = load_model(PRETRAINED_MODEL_FILE)
    return model, pretrained_model

def upload_to_gcs(image: Image, filename: str) -> bool:
    """Uploads an image to Google Cloud Storage."""
    try:
        blob = bucket.blob(filename)
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        blob.upload_from_file(image_buffer, content_type='image/png')
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False
    
class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )

    def __call__(self, img_array, cls):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            loss = predictions[:, cls]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.zeros(output.shape[0:2], dtype=np.float32)

        for index, w in enumerate(weights):
            cam += w * output[:, :, index]

        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam

def apply_heatmap(img, heatmap, heatmap_ratio=0.6):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(heatmap * heatmap_ratio + img * (1 - heatmap_ratio))

def load_image(img_path, df, preprocess=True, H=320, W=320):
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    x = image.img_to_array(x)
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image Index"].values):
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))
    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def compute_gradcam(img, model, df, labels, layer_name='bn'):
    preprocessed_input = load_image(img, df)
    predictions = model.predict(preprocessed_input)

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_labels = [labels[i] for i in top_indices]
    top_predictions = [predictions[0][i] for i in top_indices]

    original_image = load_image(img, df, preprocess=False)

    grad_cam = GradCAM(model, layer_name)

    gradcam_images = []
    for i in range(3):
        idx = top_indices[i]
        label = top_labels[i]
        prob = top_predictions[i]

        gradcam = grad_cam(preprocessed_input, idx)
        gradcam_image = apply_heatmap(original_image, gradcam)
        gradcam_images.append((gradcam_image, f"{label}: p={prob:.3f}"))

    return gradcam_images

def gradio_interface_gradcam(img, df, labels, model_path, pretrained_model_path, layer_name='bn'):
    model, pretrained_model = load_model_from_gcs() 
    labels = labels.split(',')
    gradcam_images = compute_gradcam(img, pretrained_model, df, labels, layer_name)
    return [gr.Image.update(value=image, label=label) for image, label in gradcam_images]

def calculate_mse(original_image, enhanced_image):
    mse = np.mean((original_image - enhanced_image) ** 2)
    return mse

def calculate_psnr(original_image, enhanced_image):
    mse = calculate_mse(original_image, enhanced_image)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_maxerr(original_image, enhanced_image):
    maxerr = np.max((original_image - enhanced_image) ** 2)
    return maxerr

def calculate_l2rat(original_image, enhanced_image):
    l2norm_ratio = np.sum(original_image ** 2) / np.sum((original_image - enhanced_image) ** 2)
    return l2norm_ratio

def process_image(original_image, enhancement_type, fix_monochrome=True):
    if fix_monochrome:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    image = original_image - np.min(original_image)
    image = image / np.max(original_image)
    image = (image * 255).astype(np.uint8)

    enhanced_image = enhance_image(image, enhancement_type)

    mse = calculate_mse(original_image, enhanced_image)
    psnr = calculate_psnr(original_image, enhanced_image)
    maxerr = calculate_maxerr(original_image, enhanced_image)
    l2rat = calculate_l2rat(original_image, enhanced_image)

    return enhanced_image, mse, psnr, maxerr, l2rat

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def invert(image):
    return cv2.bitwise_not(image)

def hp_filter(image, kernel=None):
    if kernel is None:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def unsharp_mask(image, radius=5, amount=2):
    def usm(image, radius, amount):
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened
    return usm(image, radius, amount)

def hist_eq(image):
    return cv2.equalizeHist(image)

def enhance_image(image, enhancement_type):
    if enhancement_type == "invert":
        return invert(image)
    elif enhancement_type == "hp_filter":
        return hp_filter(image)
    elif enhancement_type == "unsharp_mask":
        return unsharp_mask(image)
    elif enhancement_type == "hist_eq":
        return hist_eq(image)
    elif enhancement_type == "clahe":
        return apply_clahe(image)
    else:
        raise ValueError(f"Unknown enhancement type: {enhancement_type}")
    
    
app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        """Processes an uploaded image and returns the enhanced image and metrics."""
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in image_file.filename or \
                image_file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid image file'}), 400
    
        try:
            # Open the image using Pillow
            image = Image.open(image_file).convert('RGB') 

            # Convert to NumPy array
            image_np = np.array(image)

            # Get enhancement_type from request (e.g., 'CLAHE', 'Invert', etc.)
            enhancement_type = request.form.get('enhancement_type')
            if not enhancement_type:
                return jsonify({'error': 'Missing enhancement_type parameter'}), 400

            # Apply image processing
            enhanced_image, mse, psnr, maxerr, l2rat = process_image(image_np, enhancement_type)

            # Convert processed image back to PIL format for saving
            enhanced_image_pil = Image.fromarray(enhanced_image)

            # Save to in-memory buffer
            image_buffer = io.BytesIO()
            enhanced_image_pil.save(image_buffer, format='PNG') 
            image_buffer.seek(0)

            # Encode to base64
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

            return render_template('index.html', 
                image_base64=image_base64, 
                mse=float(mse), 
                psnr=float(psnr), 
                maxerr=float(maxerr), 
                l2rat=float(l2rat)) 
            
            # Create unique folder name
            folder_name = str(uuid.uuid4())

            # Create the folder in Cloud Storage
            bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')

            if upload_to_gcs(image, folder_name + '/' + 'original_image'):
                print(f"Original image uploaded to Cloud Storage: gs://{BUCKET_NAME}/{folder_name}/original_image")
            else:
                print("Error uploading original image to Cloud Storage.")

            # Upload processed image to the same folder
            if upload_to_gcs(enhanced_image_pil, folder_name + '/' + enhancement_type):
                print(f"Processed image uploaded to Cloud Storage: gs://{BUCKET_NAME}/{folder_name}/{enhancement_type}")
            else:
                print("Error uploading processed image to Cloud Storage.")
                    
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    else:
        return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_api():
    """Processes an uploaded image and returns the enhanced image and metrics."""
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in image_file.filename or image_file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid image file'}), 400
    
    try:
        # Open the image using Pillow
        image = Image.open(image_file).convert('RGB') 

        # Convert to NumPy array
        image_np = np.array(image)

        # Get enhancement_type from request (e.g., 'CLAHE', 'Invert', etc.)
        enhancement_type = request.form.get('enhancement_type')
        if not enhancement_type:
            return jsonify({'error': 'Missing enhancement_type parameter'}), 400

        # Apply image processing
        enhanced_image, mse, psnr, maxerr, l2rat = process_image(image_np, enhancement_type)

        # Convert processed image back to PIL format for saving
        enhanced_image_pil = Image.fromarray(enhanced_image)

        # Save to in-memory buffer
        image_buffer = io.BytesIO()
        enhanced_image_pil.save(image_buffer, format='PNG') 
        image_buffer.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        response = {
        'message': 'Image processed successfully!',
        'processed_image': image_base64,
        'mse': float(mse), 
        'psnr': float(psnr), 
        'maxerr': float(maxerr),  
        'l2rat': float(l2rat)   
        }   
        # Create unique folder name
        folder_name = str(uuid.uuid4())

        # Create the folder in Cloud Storage
        bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')

        if upload_to_gcs(image, folder_name + '/' + 'original_image'):
            print(f"Original image uploaded to Cloud Storage: gs://{BUCKET_NAME}/{folder_name}/original_image")
        else:
            print("Error uploading original image to Cloud Storage.")

        # Upload processed image to the same folder
        if upload_to_gcs(enhanced_image_pil, folder_name + '/' + enhancement_type):
            print(f"Processed image uploaded to Cloud Storage: gs://{BUCKET_NAME}/{folder_name}/{enhancement_type}")
        else:
            print("Error uploading processed image to Cloud Storage.")
                
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)