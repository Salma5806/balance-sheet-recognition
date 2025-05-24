from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
from paddleocr import PaddleOCR
from PIL import Image
import torch
import numpy as np
import re
import pandas as pd
import gdown

def unnormalize_box(box, width, height):
    return [
        int(box[0] / 1000 * width),
        int(box[1] / 1000 * height),
        int(box[2] / 1000 * width),
        int(box[3] / 1000 * height)
    ]

def extract_year(text):
    text = text.lower().replace('–', '-')

    # Exemples directs : 2023, 2024...
    match = re.search(r'\b(20[2-3][0-9])\b', text)
    if match:
        return match.group(1)

    # Formats comme 31/12/2023, 31-12-2023, 31.12.2023, 31 décembre 2022
    match = re.search(r'(31[^\d]?(?:12|déc)[^\d]?(20[2-3][0-9]))', text)
    if match:
        return match.group(2)

    # Format court : 31-déc.-23
    match = re.search(r'31[^\d]?(?:12|déc)[^\d]?([0-9]{2})', text)
    if match:
        return '20' + match.group(1)

    # Format 2023R, 2024P, etc.
    match = re.search(r'\b(20[2-3][0-9])[RP]?\b', text)
    if match:
        return match.group(1)
    
    match =  re.search(r'31[^\d]?(?:12|déc)[^\d]?([0-9]{2})', text)
    if match:
        return '20' + match.group(1)

    return None
model_path = gdown.download("https://drive.google.com/drive/folders/1TnwJKw1dwi0XcJLTWe91Pikof4GX4b6T?usp=drive_link")
def predict_labels(image_path, model=model_path):
    processor = LayoutLMv3Processor.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
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

    # Supprimer les doublons
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

    # Détection des années avec position x
    year_positions = {}
    for item in year_items:
        year = extract_year(item['text'])
        if year and year not in year_positions:
            year_positions[year] = item['box'][0]
            print(f"Année détectée : {year}, Position x : {item['box'][0]}")

    # Création du tableau final
    data = []
    for key_item in key_items:
        key_text = key_item['text']
        key_box = key_item['box']
        key_y_min = key_box[1]

        # Chercher la note associée (facultatif)
        note = ''
        for name_item in name_items:
            if 'CP-' in name_item['text'] or 'p-' in name_item['text']:
                name_y_min = name_item['box'][1]
                if abs(name_y_min - key_y_min) < 10:
                    note = name_item['text']
                    break

        # Associer chaque année à sa valeur la plus proche (en x et y)
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
        for year in sorted(year_positions.keys(), reverse=True):  # 2023, 2022, ...
            row[year] = year_values.get(year, 0)
        data.append(row)

    df = pd.DataFrame(data)
    return {
        "image": image,
        "true_predictions": true_predictions,
        "true_boxes": true_boxes,
        "df": df
    }
