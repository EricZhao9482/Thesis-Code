from manga_ocr import MangaOcr
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize OCR
mocr = MangaOcr()

# Initialize Translator
model_name = 'Helsinki-NLP/opus-mt-ja-en'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

max_len_limit = 1000
# normal panel
text = mocr("mangapanel.png")
print(f"uncropped text: {text}")

trns_text = translator(text, max_length=max_len_limit)
print(f"trns_uncroptext{trns_text}")

# cropped panel
croptext = mocr("mangapanelcrop.png")
print(f"normal_text: {croptext}")

trns_croptext = translator(croptext[:-1], max_length=max_len_limit)
print(f"trns_croptext: {trns_croptext}")