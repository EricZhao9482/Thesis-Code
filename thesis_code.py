import cv2
import pytesseract
from manga_ocr import MangaOcr
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pyuac
from torch.utils.data import Dataset, DataLoader
import importlib
import sys
sys.path.append("C:\\Users\\ericz\\Documents\\Capstone\\Senior Thesis\\Thesis Code\\BLIP2-Japanese-master")
from lavis.common.registry import registry
from lavis.models import model_zoo, load_model_and_preprocess
from PIL import Image
import os

def main():

    # # set tesseract path for ocr
    # pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR"

    # Initialize OCR
    mocr = MangaOcr()

    # Initialize Translator
    model_name = 'Helsinki-NLP/opus-mt-ja-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    
    # Set up for image captioning software 
    # from BLIP2_Japanese_master.lavis.models import model_zoo, load_model_and_preprocess
    # BJMLavisModels = "BLIP2-Japanese-master.lavis.models"
    # ImageCaptioning = importlib.import_module(BJMLavisModels)
    print(model_zoo)
    model, visual_encoder, text_encoder = load_model_and_preprocess('blip2_Japanese', 'finetune') #load_model_and_preprocess('blip2_Japanese', 'pretrain')
    

    # Set up image for later OCR
    img = cv2.imread("mangapanel.png")

    # Preprocessing the image starts
    
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size. 
    # Kernel size increases or decreases the area 
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect 
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = img.copy()

    file = open("recognized.txt", "w+")
    file.write("")
    file.close()
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        
        # Open the file in append mode
        file = open("recognized.txt", "a")
        
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        
        # Appending the text into file
        file.write(text)
        file.write("\n")
        
        # Close the file
        file.close

    test_panel = "mangapanel.png"

    # print(f"test caption: {model.generate(test_panel)}")

    max_len_limit = 1000
    # normal panel
    text = mocr(test_panel)
    print(f"uncropped text: {text}")

    trns_text = translator(text, max_length=max_len_limit)
    print(f"trns_uncroptext{trns_text}")

    # cropped panel
    croptext = mocr("mangapanelcrop.png")
    print(f"normal_text: {croptext}")

    trns_croptext = translator(croptext[:-1], max_length=max_len_limit)
    print(f"trns_croptext: {trns_croptext}")
if __name__ == "__main__":
    # if not pyuac.isUserAdmin():
    #     print("Re-launching as admin!")
    #     pyuac.runAsAdmin()
    # else:        
    main()  # Already an admin here.