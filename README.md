# VocalBridge Backend 

This is the backend server for **VocalBridge**, an AI-powered English–Nepali translation and voice bridge system.  
It provides real-time translation between English and Nepali using a sequence-to-sequence neural network with attention, built using **PyTorch** and **Flask**.
---

### **This repository contains **only the backend** for VocalBridge. The frontend / user interface is developed separately at https://github.com/Szhoosh/VocalBridgeFrontend and not included here.️**


             
#  Features

- **Real-Time Voice Cloning**
  - Clone a voice using just a few seconds of audio.
  - Generate speech with the cloned voice from text.

-  **Speech Translation**
  - Translate speech from one language to another using neural translation models.
  - Supports **multi-language transliteration**.

-  **Text-to-Speech (TTS)**
  - Natural-sounding speech Clonned generation powered by `pyttsx3` and custom synthesizer models.

-  **Audio Processing**
  - Audio loading, normalization, feature extraction via `librosa` and `soundfile`.

-  **Image-to-Text (OCR) Support**
  - Extract text from images with `easyocr` for visual translation.

-  **Language Detection**
  - Automatically detect spoken or written language with `langdetect` and `pycountry`.
#  Highlights

- Neural machine translation (English ↔ Nepali)
- Flask-based API for frontend integration
- Pretrained model loading (`PT-eng-nep-V1.pt`)
- Language preprocessing and normalization pipeline
- Simple frontend (`index.html`) for testing
- Modular architecture with `Server.py` (API) and `translator.py` (model logic)



---



## Example Workflow

Record a short voice sample.

Encode it with the encoder.

Generate speech using synthesizer + vocoder.

Translate output text or speech using translation module.

Optionally, use OCR for text extraction from images.
