import json
from keras.models import Model, load_model
import gradio as gr
import cv2

# Load the model without compilation
model = load_model('final_vgg1920epochs.h5', compile=False)

# Open JSON file
with open('dat.json') as f:
    data = json.load(f)

keys = list(data)

def Predict(image):
    img = cv2.resize(image, (32, 32)) / 255.0
    prediction = model.predict(img.reshape(1, 32, 32, 3))
    
    return (keys[prediction.argmax()],
            data[keys[prediction.argmax()]]['description'],
            data[keys[prediction.argmax()]]['symptoms'],
            data[keys[prediction.argmax()]]['causes'],
            data[keys[prediction.argmax()]]['treatement-1'])

# Update Gradio interface
app = gr.Interface(
    fn=Predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label='Name Of Disease'),
        gr.Textbox(label='Description'),
        gr.Textbox(label='Symptoms'),
        gr.Textbox(label='Causes'),
        gr.Textbox(label='Treatment')
    ],
    title="Skin Disease Classification",
    description='This Space predicts these diseases:\n\n'
                '1) Acne and Rosacea Photos.\n'
                '2) Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions.\n'
                '3) Eczema Photos.\n'
                '4) Melanoma Skin Cancer, Nevi, and Moles.\n'
                '5) Psoriasis pictures, Lichen Planus, and related diseases.\n'
                '6) Tinea Ringworm, Candidiasis, and other Fungal Infections.\n'
                '7) Urticaria (Hives).\n'
                '8) Nail Fungus and other Nail Diseases.\n'
)

# Launch the Gradio app
app.launch(debug=True)
