# VocalBridge Backend 

This is the backend server for **VocalBridge**, an AI-powered Language translation and voice bridge system.  
It provides real-time translation between English and Nepali using a sequence-to-sequence neural network with attention, built using **PyTorch** and **Flask**.

---

##  Features

- Neural machine translation (English ↔ Nepali)
- Flask-based API for frontend integration
- Pretrained model loading (`PT-eng-nep-V1.pt`)
- Language preprocessing and normalization pipeline
- Simple frontend (`index.html`) for testing
- Modular architecture with `Server.py` (API) and `translator.py` (model logic)

---

## Model Details

- **Architecture:** Sequence-to-sequence GRU with Bahdanau Attention
- **Input:** English sentences (tokenized, normalized)
- **Output:** Nepali translation tokens
- **Model file:** `PT-eng-nep-V1.pt`
- **Max sequence length:** 10

---
