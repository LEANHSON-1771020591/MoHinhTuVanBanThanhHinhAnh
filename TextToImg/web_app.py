from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline
from datetime import datetime
import os
import torch
from authtoken import auth_token

app = Flask(__name__)

# Cấu hình thư mục lưu ảnh
OUTPUT_DIR = os.path.join('static', 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Khởi tạo pipeline với cấu hình tối ưu cho VRAM 4GB
def load_pipeline():
    # Sử dụng Stable Diffusion 1.5 từ RunwayML
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Initialize pipeline with CPU optimizations
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use full precision for CPU
        use_auth_token=auth_token,
        requires_safety_checker=False,
        safety_checker=None
    ).to("cpu")
    
    # Enable memory efficient features for CPU
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    
    return pipeline

# Khởi tạo pipeline
pipeline = load_pipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Lấy thông tin từ form
        prompt = request.form['prompt']
        num_steps = int(request.form.get('num_steps', 50))  # Tăng số bước mặc định để cải thiện chất lượng
        guidance_scale = float(request.form.get('guidance_scale', 9.0))  # Tăng guidance scale để ảnh rõ nét hơn
        
        # Tạo ảnh với kích thước nhỏ hơn để tiết kiệm bộ nhớ
        image = pipeline(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=512,  # Tăng kích thước để chất lượng tốt hơn
            width=512,   # Tăng kích thước để chất lượng tốt hơn
        ).images[0]
        
        # Lưu ảnh
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f'generated_{timestamp}.png')
        image.save(output_path)
        
        return {'success': True, 'image_path': output_path}
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)