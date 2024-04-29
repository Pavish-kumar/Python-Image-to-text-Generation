!pip install gradio
!pip install transformers
!pip install Image
!pip install gtts


import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from gtts import gTTS
import IPython.display as ipd

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

def caption_image(img):
    # Convert the input image to RGB format
    raw_image = Image.fromarray(img).convert('RGB')

    # Conditional image captioning
    text = "A photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_audio(caption):
    tts = gTTS(caption, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file

def caption_and_generate_audio(img):
    caption = caption_image(img)
    audio_file = generate_audio(caption)
    return caption.capitalize(), audio_file

iface = gr.Interface(fn=caption_and_generate_audio, inputs="image", outputs=["text", "audio"])
iface.launch()