from flask import Flask, request, jsonify
import cv2, json, torch, pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# 🔹 Load models and label mapping
model_path = 'bert_model_last'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = json.load(open(f'{model_path}/label_mapping.json'))
reverse_map = {v: k for k, v in label_map.items()}

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model.eval()

med_db_csv = "medicine_db.csv"
med_db = pd.read_csv(med_db_csv)

# 🔹 Utilities
def preprocess_image_for_ocr(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_image = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    return pil_image

def trocr_ocr_from_crop(crop_np):
    pil_img = preprocess_image_for_ocr(crop_np)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip().lower()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_idx = torch.argmax(outputs.logits, dim=1).item()
    return reverse_map[pred_idx]

def extract_line_crops(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_imgs = []
    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20 and w > 50:
            crop = image[y:y+h, x:x+w]
            cropped_imgs.append(crop)
    return cropped_imgs

# 🔹 API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_np = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    crops = extract_line_crops(image_np)
    results = []

    for crop in crops:
        text = trocr_ocr_from_crop(crop)
        if len(text) > 2:
            pred = classify_text(text)
            match = med_db[med_db['medicine_name'].str.lower() == pred.lower()]
            generic = match.iloc[0]['generic_name'] if not match.empty else "Not found"
            results.append({
                "Extracted": text,
                "Predicted": pred,
                "Generic": generic
            })

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
