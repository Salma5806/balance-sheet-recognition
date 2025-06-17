import os
from pdf2image import convert_from_path
import pytesseract

# 🛠️ Chemin vers ton fichier PDF
pdf_path = "ton_fichier.pdf"

# 📁 Dossier temporaire pour les images
temp_img_dir = "temp_images"
os.makedirs(temp_img_dir, exist_ok=True)

# 🔤 Langue OCR (changer 'fra' en 'eng' si c'est de l'anglais)
ocr_lang = 'fra'  # ou 'eng', 'ara', etc.

# 🔍 Fonction principale
def extract_text_from_scanned_pdf(pdf_path, lang='fra'):
    try:
        # 📄 Convertir chaque page en image (nécessite Poppler)
        images = convert_from_path(pdf_path, dpi=300)

        all_text = ""
        for i, img in enumerate(images):
            img_path = os.path.join(temp_img_dir, f"page_{i+1}.png")
            img.save(img_path, 'PNG')

            # 🔠 OCR avec pytesseract
            text = pytesseract.image_to_string(img, lang=lang)
            print(f"\n📝 Texte page {i+1}:\n{text}")
            all_text += f"\n--- Page {i+1} ---\n{text}"

        return all_text

    except Exception as e:
        print(f"❌ Erreur lors du traitement du PDF : {e}")
        return None

# ▶️ Lancer la fonction
if __name__ == "__main__":
    result = extract_text_from_scanned_pdf(pdf_path, lang=ocr_lang)

    if result:
        with open("texte_extrait.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("\n✅ Extraction terminée. Le texte a été sauvegardé dans 'texte_extrait.txt'.")
