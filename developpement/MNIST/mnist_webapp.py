import gradio as gr
from PIL import Image
import requests
import io
import numpy as np


def recognize_digit(image):
    # Check if `image` is a NumPy array or a PIL Image
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image if necessary
        image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://127.0.0.1:5001/predict", data=img_binary.getvalue())
    return response.json()["prediction"]
 
if __name__=='__main__':
 
    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);