import os
import torch
import atexit  
import shutil
import imageio
from PIL import Image
from datetime import datetime
import segmentation_models_pytorch as smp
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for, session
from functions import load_image, prediction, visualize_image, normalize_image, visualize_prediction



app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "jhaofuyl;52668"


app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['VIS_FOLDER'] = 'static/vis'


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3(encoder_name="resnet50", encoder_weights=None, in_channels=12, classes=1)  # Load model architecture
state_dict = torch.load('deeplabv3.pth', weights_only=True)  # Load model weights
model.load_state_dict(state_dict)
model.eval()


@app.route('/',)
def index():
    # Clear the session flag if the page is refreshed (GET request)
    session.pop("form_submitted", None)
    return render_template("index.html")



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"
    
    file = request.files['image']

    if file.filename == '':
        return "No file selected"
    
    # Save the Uploaded image
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
   

    # Load the image
    image = load_image(file_path)

    # Normalize the image
    image = normalize_image(image)

    # Visualize the image
    image_vis = visualize_image(file_path)
    
    # Save the visualization
    vis_filename = os.path.splitext(filename)[0] + ".png"
    vis_path = os.path.join(app.config['VIS_FOLDER'], vis_filename)
    vis_image = Image.fromarray(image_vis)
    vis_image.save(vis_path, format="PNG")

    # Make prediction
    pred = prediction(model, image, device)

    # visualize the prediction
    pred_vis = visualize_prediction(pred)

    # Save the prediction
    out_filename = os.path.splitext(filename)[0] + ".png"
    output_path  = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
    pred_vis.save(output_path, format="PNG")

    # Add a timestamp to the image URLs to prevent caching
    timestamp = int(datetime.timestamp(datetime.now()))
    uploaded_image_url = url_for('static', filename='vis/' + vis_filename) + f"?t={timestamp}"
    prediction_image_url = url_for('static', filename='outputs/' + out_filename) + f"?t={timestamp}"

    # Set session flag to indicate form submission
    session["form_submitted"] = True

   

    return render_template("index.html",uploaded_image_url=url_for('static', filename='vis/' + vis_filename), prediction_image_url=url_for('static', filename='outputs/' + out_filename), form_submitted=session.get("form_submitted", False))



def cleanup_folders():
   
    folders_to_clean = [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['VIS_FOLDER']]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory and its contents
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# Register cleanup function to run on exit
atexit.register(cleanup_folders)


if __name__ == '__main__':
    app.run(debug=True)