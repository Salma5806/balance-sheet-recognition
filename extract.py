from predict import predict_labels, unnormalize_box
import pandas as pd
import re

# Fonction pour extraire l'année à partir de différents formats
def extract_year(text):
    text = text.lower().replace('–', '-')

    patterns = [
        r"au\s*31[^\d]?(?:12|déc)[^\d]?(20[2-3][0-9])",   # au 31/12/2023
        r"31[^\d]?(?:12|déc)[^\d]?(20[2-3][0-9])",         # 31/12/2023
        r"31[^\d]?(?:12|déc)[^\d]?([0-9]{2})",           # 31-déc.-23
        r"\b(20[2-3][0-9])\b",                               # 2023, 2024
        r"\b(20[2-3][0-9])[RP]?\b",                          # 2023R
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            year = match.group(1)
            return '20' + year if len(year) == 2 else year

    return None

# Fonction principale
def extract_information(image_path):
    results_raw = predict_labels(image_path)

    required_keys = [
        'words', 'predictions', 'token_boxes',
        'image_width', 'image_height',
        'processor', 'encoding', 'model'
    ]
    for key in required_keys:
        if key not in results_raw:
            raise KeyError(f"Clé manquante dans le résultat de predict_labels : '{key}'")

    words = results_raw['words']
    predictions = results_raw['predictions']
    token_boxes = results_raw['token_boxes']
    image_width = results_raw['image_width']
    image_height = results_raw['image_height']
    processor = results_raw['processor']
    encoding = results_raw['encoding']
    model = results_raw['model']

    tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze())
    word_ids = encoding.word_ids()

    results = []
    for idx, (pred, box) in enumerate(zip(predictions, token_boxes)):
        if word_ids[idx] is None:
            continue
        word = words[word_ids[idx]]
        label = model.config.id2label[pred]
        results.append({
            'text': word,
            'label': label.lower(),
            'box': unnormalize_box(box, image_width, image_height)
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
    key_items = [item for item in unique_results if item['label'] == 'key']
    value_items = [item for item in unique_results if item['label'] == 'value']

    year_positions = {}
    for item in year_items:
        year = extract_year(item['text'])
        if year and year not in year_positions:
            year_positions[year] = item['box'][0]
            print(f"✅ Année détectée : {year} → Position X : {item['box'][0]}")

    data = []
    for key_item in key_items:
        key_text = key_item['text']
        key_box = key_item['box']
        key_y = key_box[1]

        note = ''
        for name_item in name_items:
            name_y = name_item['box'][1]
            if 'cp-' in name_item['text'].lower() or 'p-' in name_item['text'].lower():
                if abs(name_y - key_y) < 10:
                    note = name_item['text']
                    break

        year_values = {}
        for value_item in value_items:
            value_y = value_item['box'][1]
            if abs(value_y - key_y) < 10:
                for year, x_pos in year_positions.items():
                    if abs(value_item['box'][0] - x_pos) < 50:
                        value = value_item['text'].replace(' ', '').replace(',', '.')
                        year_values[year] = value
                        break

        row = {'key': key_text}
        for year in sorted(year_positions.keys(), reverse=True):
            row[year] = year_values.get(year, 0)
        data.append(row)

    df = pd.DataFrame(data)
    return df
