# Deep Learning Project

This project demonstrates a deep learning model for image classification using a Convolutional Neural Network (CNN). The project includes a Flask web application for serving the model and a Jupyter notebook for training the model.

## Directory Structure

- `flask_app.py`: Flask web application for serving the trained model.
- `cnn-image-classification.ipynb`: Jupyter notebook for training the CNN model.
- `requirements.txt`: List of dependencies required for the project.

## Flask Application

The Flask application (`flask_app.py`) provides an API for predicting the class of an image. It supports image input via file upload or URL.

### Endpoints

- `/`: Renders the index page.
- `/predict`: Accepts POST requests with an image file or URL and returns the predicted class and confidence score.

### Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask application:
   ```bash
   python flask_app.py
   ```
3. Access the application at `http://localhost:5001`.

## Jupyter Notebook

The Jupyter notebook (`cnn-image-classification.ipynb`) contains the code for training the CNN model on the CIFAR-10 dataset. It includes data preprocessing, model architecture, training, and evaluation.

### Steps

1. Load and preprocess the CIFAR-10 dataset.
2. Define the CNN model architecture.
3. Train the model with data augmentation.
4. Evaluate the model on the test dataset.
5. Save the trained model.

## Dependencies

The project requires the following dependencies, which are listed in `requirements.txt`:

Install the dependencies using:

```bash
pip install -r requirements.txt
```
