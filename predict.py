from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
from paddleocr import PaddleOCR
from PIL import Image
import torch
import numpy as np
import re
import pandas as pd
import zipfile
import os

def download_model():
    model_dir = "final_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Google Drive file IDs (replace with your actual IDs)
    files = {
        "vocab.json": "https://drive.google.com/uc?id=YOUR_VOCAB_ID",
        "training_args.bin": "https://drive.google.com/uc?id=YOUR_TRAINING_ARGS_ID",
        "tokenizer.json": "https://drive.google.com/uc?id=YOUR_TOKENIZER_JSON_ID",
        "tokenizer_config.json": "https://drive.google.com/uc?id=YOUR_TOKENIZER_CONFIG_ID",
        "special_tokens_map.json": "https://drive.google.com/uc?id=YOUR_SPECIAL_TOKENS_ID",
        "preprocessor_config.json": "https://drive.google.com/uc?id=YOUR_PREPROCESSOR_ID",
        "model.safetensors": "https://drive.google.com/uc?id=YOUR_MODEL_ID",
        "merges.txt": "https://drive.google.com/uc?id=YOUR_MERGES_ID",
        "config.json": "https://drive.google.com/uc?id=YOUR_CONFIG_ID"
    }
    
    # Download missing files
    for filename, url in files.items():
        if not os.path.exists(os.path.join(model_dir, filename)):
            gdown.download(url, os.path.join(model_dir, filename), quiet=False)
    
    return model_dir


def predict_labels(image_path):
   model_dir = download_model()
    
    # Load processor and model
    processor = LayoutLMv3Processor.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    ocr = PaddleOCR(use_angle_cls=False, lang='fr', rec=False)
    result = ocr.ocr(image_np, cls=False)

    words = []
    boxes = []

    for item in result[0]:
        box = item[0]
        text = item[1][0]
        if text.strip() == "":
            continue
        words.append(text)
        x_min = min([pt[0] for pt in box])
        y_min = min([pt[1] for pt in box])
        x_max = max([pt[0] for pt in box])
        y_max = max([pt[1] for pt in box])
        norm_box = [
            int((x_min / w) * 1000),
            int((y_min / h) * 1000),
            int((x_max / w) * 1000),
            int((y_max / h) * 1000)
        ]
        boxes.append(norm_box)

    encoding = processor(images=image, text=words, boxes=boxes, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    true_predictions = [model.config.id2label[pred] for pred in predictions]
    true_boxes = [unnormalize_box(box, w, h) for box in token_boxes]

    tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze())
    word_ids = encoding.word_ids()

    results = []
    for idx, (pred, box) in enumerate(zip(predictions, token_boxes)):
        label = model.config.id2label[pred]
        if word_ids[idx] is None:
            continue
        word = words[word_ids[idx]]
        results.append({
            'text': word,
            'label': label.lower(),
            'box': unnormalize_box(box, w, h)
        })

    unique_results = []
    seen = set()
    for item in results:
        key = (item['text'], item['label'], tuple(item['box']))
        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    name_items = [item for item in unique_results if item['label'] == 'name']
    year_items = [item for item in unique_results if item['label'] == 'year']
    reference_items = [item for item in unique_results if item['label'] == 'reference']
    key_items = sorted([item for item in unique_results if item['label'] == 'key'], key=lambda x: x['box'][1])
    value_items = [item for item in unique_results if item['label'] == 'value']

    year_positions = {}
    for item in year_items:
        year = extract_year(item['text'])
        if year and year not in year_positions:
            year_positions[year] = item['box'][0]
            print(f"Année détectée : {year}, Position x : {item['box'][0]}")

    data = []
    for key_item in key_items:
        key_text = key_item['text']
        key_box = key_item['box']
        key_y_min = key_box[1]

        note = ''
        for name_item in name_items:
            if 'CP-' in name_item['text'] or 'p-' in name_item['text']:
                name_y_min = name_item['box'][1]
                if abs(name_y_min - key_y_min) < 10:
                    note = name_item['text']
                    break

        year_values = {}
        for value_item in value_items:
            value_box = value_item['box']
            value_y_min = value_box[1]
            if abs(value_y_min - key_y_min) < 10:
                for year, x_pos in year_positions.items():
                    if abs(value_box[0] - x_pos) < 50:
                        value = value_item['text'].replace(' ', '').replace(',', '.')
                        year_values[year] = value
                        break

        row = {
            'key': key_text
        }
        for year in sorted(year_positions.keys(), reverse=True):
            row[year] = year_values.get(year, 0)
        data.append(row)

    df = pd.DataFrame(data)
    return {
        "image": image,
        "true_predictions": true_predictions,
        "true_boxes": true_boxes,
        "df": df
    }
