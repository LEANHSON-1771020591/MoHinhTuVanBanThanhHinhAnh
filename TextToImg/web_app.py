from flask import Flask, render_template, request, send_from_directory
import torch
import gc
import datetime
import os
from diffusers import StableDiffusionPipeline
from PIL import Image
from authtoken import auth_token

app = Flask(__name__)

# Cấu hình GPU
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình
model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=auth_token
).to(device)

model.enable_attention_slicing()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('prompt').strip()
        if not text:
            return render_template('index.html', error="Vui lòng nhập một prompt.")
        
        start_time = datetime.datetime.now()
        guidance_scale = float(request.form.get('guidance_scale', 7.0))
        num_inference_steps = int(request.form.get('num_inference_steps', 20))
        
        with torch.inference_mode():
            image = model(text, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Lưu ảnh
        if not os.path.exists('static/output'):
            os.makedirs('static/output')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f'static/output/generated_{timestamp}.png'
        image.save(image_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return render_template('index.html', image_path=image_path, processing_time=processing_time)
    
    return render_template('index.html')

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory('static/output', filename)

if __name__ == '__main__':
    app.run(debug=True)