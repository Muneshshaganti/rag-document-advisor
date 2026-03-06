from pdf2image import convert_from_path
import pytesseract

PDF_PATH = "NATURES_AMODHA_DOCUMENT.pdf"

print("Starting OCR... this may take several minutes.")

pages = convert_from_path(PDF_PATH)

text = ""

for i, page in enumerate(pages):
    print(f"Processing page {i+1}/{len(pages)}")
    text += pytesseract.image_to_string(page)

with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("OCR finished. Text saved to extracted_text.txt")
