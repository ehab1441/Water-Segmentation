# Water Segmentation Application

## Overview

This application is designed to perform water segmentation on satellite images using a deep learning model. It leverages the DeepLabV3 architecture with a ResNet50 encoder to segment water regions in images. The application is built using Flask for the backend and Bootstrap for the frontend, providing a user-friendly interface for uploading images and viewing segmentation results.

## Features

- **Image Upload**: Users can upload satellite images for water segmentation.
- **Segmentation Prediction**: The application processes the uploaded image using a pre-trained DeepLabV3 model to predict water regions.
- **Visualization**: Both the uploaded image and the segmentation result are displayed side by side for easy comparison.
- **Responsive Design**: The interface is designed to be responsive and works well on different screen sizes.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ehab1441/Water-Segmentation.git
   cd water-segmentation-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-trained model weights**:
   - Ensure that the `deeplabv3.pth` file is placed in the root directory of the project.

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   - Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage

1. **Upload an Image**:
   - Click on the "Upload Image" button to select a satellite image from your local machine.

2. **Predict**:
   - Click the "Predict" button to process the image. The application will display the uploaded image and the segmentation result side by side.

3. **View Results**:
   - The uploaded image and the predicted water segmentation will be displayed in the "Result Section".

---

### The Model, Training, and Evaluation

To explore the model architecture, training process, and evaluation metrics, please refer to the **`water_segmentation.ipynb`** notebook. The notebook contains detailed code and explanations for:

1. **Model Architecture**:
   - The notebook uses the **DeepLabV3** model with a **ResNet50** encoder for semantic segmentation.
   - The model is designed to handle 12 input channels (spectral bands) and outputs a binary segmentation mask (water vs. non-water).

2. **Training Process**:
   - The model is trained using **Binary Cross-Entropy Loss with Logits (BCEWithLogitsLoss)** and the **Adam optimizer**.
   - The training loop includes forward passes, loss calculation, backward passes, and model saving based on validation loss.

3. **Evaluation**:
   - The model's performance is evaluated using metrics such as **Intersection over Union (IoU)**, **Precision**, **Recall**, and **F1-Score**.
   - Predictions are visualized to qualitatively assess the model's performance.

4. **Dataset Visualization**:
   - The notebook includes code to visualize the input images and their corresponding masks, ensuring that the dataset is correctly loaded and preprocessed.

To view the notebook:
1. Open the **`water_segmentation.ipynb`** file in Jupyter Notebook or any compatible environment.

---

## File Structure

```
water-segmentation-app/
│
├── app.py                # Main Flask application
├── deeplabv3.pth         # Pre-trained model weights
├── functions.py          # Helper functions for image processing
├── requirements.txt      # List of dependencies
├── water_segmentation.ipynb  # Notebook for model training and evaluation
├── static/               # Static files (CSS, JS, images)
│   ├── styles.css        # Custom CSS
│   ├── scripts.js        # Custom JS
│   ├── uploads/          # Folder for uploaded images
│   ├── outputs/          # Folder for output images (predictions)
│   └── vis/              # Folder for visualized images
├── templates/            # HTML templates
│   └── index.html        # Main HTML template
└── README.md             # This file

```

## Dependencies

- Flask
- PyTorch
- Segmentation Models PyTorch (smp)
- Pillow (PIL)
- Werkzeug
- Imageio
- Bootstrap (via CDN)

## Cleanup

The application automatically cleans up the uploaded and generated images when the server is stopped. This ensures that no unnecessary files are left behind.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The DeepLabV3 model architecture is provided by the [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) library.
- The Flask framework is used for building the web application.
- Bootstrap is used for the frontend design.

---

For any questions or issues, please contact Mohamed Ehab at m.ehab1441@gmail.com .