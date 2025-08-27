import argparse
import os
import io
from datetime import datetime

from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
from PIL import Image
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# --- 1. Flask 앱 및 기본 설정 ---
app = Flask(__name__)

# 스크립트가 위치한 디렉터리를 기준으로 절대 경로 생성
_root_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(_root_dir, 'static/uploads')
RESULT_FOLDER = os.path.join(_root_dir, 'static/results')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- 2. 모델 로딩 (앱 실행 시 한 번만) ---
pipeline = None
automasker = None

def load_models():
    """CatVTON 모델과 AutoMasker를 로드합니다."""
    global pipeline, automasker
    if pipeline is not None:
        return

    print("Loading CatVTON models... This may take a moment.")
    # Hugging Face에서 모델 가중치 다운로드
    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

    # CatVTON 파이프라인 초기화
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("bf16"),
        use_tf32=True,
        device='cuda'
    )
    
    # 자동 마스크 생성기 초기화
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device='cuda',
    )
    print("Models loaded successfully.")


# --- 3. 웹페이지 및 API 라우트 정의 ---

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/tryon', methods=['POST'])
def tryon():
    """이미지를 받아 가상 피팅을 수행하는 API"""
    if 'person_image' not in request.files or 'cloth_image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    person_file = request.files['person_image']
    cloth_file = request.files['cloth_image']

    if person_file.filename == '' or cloth_file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    try:
        # 이미지 열기 및 전처리
        person_image = Image.open(io.BytesIO(person_file.read())).convert("RGB")
        cloth_image = Image.open(io.BytesIO(cloth_file.read())).convert("RGB")

        width, height = 768, 1024
        person_image = resize_and_crop(person_image, (width, height))
        cloth_image = resize_and_padding(cloth_image, (width, height))

        # 자동 마스크 생성 (상체 옷 기준)
        mask = automasker(person_image, cloth_type='upper')['mask']

        # 가상 피팅 파이프라인 실행
        with torch.no_grad():
            result_image = pipeline(
                image=person_image,
                condition_image=cloth_image,
                mask=mask,
                num_inference_steps=50,
                guidance_scale=2.5,
                generator=torch.Generator(device='cuda').manual_seed(42)
            )[0]

        # 결과 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.png"
        result_save_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_image.save(result_save_path)

        return jsonify({'result_url': f'/{result_save_path}'})

    except Exception as e:
        print(f"Error during try-on: {e}")
        return jsonify({'error': '가상 피팅 중 오류가 발생했습니다.'}), 500

@app.route('/static/results/<filename>')
def send_result_image(filename):
    """생성된 결과 이미지를 제공합니다."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


# --- 4. 앱 실행 ---
if __name__ == '__main__':
    load_models()  # 앱 시작 시 모델 로드
    app.run(debug=True, host='0.0.0.0', port=5001)