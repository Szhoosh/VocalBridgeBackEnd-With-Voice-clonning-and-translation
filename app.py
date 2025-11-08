from __future__ import unicode_literals, print_function, division
from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import os
import numpy as np
import soundfile as sf
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from model import EncoderRNN, AttnDecoderRNN, BahdanauAttention, Lang
from encoder import inference as encoderv
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from toolbox.ui import UI
from toolbox.utterance import Utterance
from pathlib import Path
import librosa
from io import open
import unicodedata
import re
import random
import pyttsx3
import pyaudio
import wave
from faster_whisper import WhisperModel

import easyocr
from PIL import Image
import cv2

from langdetect import detect
import pycountry

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
devanagari_text = ''
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Romanized Text:", romanized_text)
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
special_character_mapping = {
    'ā': 'a', 'Ā': 'A',
    'ī': 'i', 'Ī': 'I',
    'ū': 'u', 'Ū': 'U',
    'ṛ': 'r', 'Ṛ': 'R',
    'ṝ': 'rr', 'Ṝ': 'RR',
    'ḷ': 'l', 'Ḷ': 'L',
    'ḹ': 'll', 'Ḹ': 'LL',
    'ṅ': 'n', 'Ṅ': 'N',
    'ñ': 'n', 'Ñ': 'N',
    'ṭ': 't', 'Ṭ': 'T',
    'ḍ': 'd', 'Ḍ': 'D',
    'ṇ': 'n', 'Ṇ': 'N',
    'ś': 's', 'Ś': 'S',
    'ṣ': 's', 'Ṣ': 'S',
    'ṃ': 'm', 'Ṃ': 'M',
    'ḥ': 'h', 'Ḥ': 'H',
    'ḻ': 'l', 'Ḻ': 'L',
    'ẏ': 'y', 'Ẏ': 'Y',
    'ǰ': 'j', 'ǲ': 'J',
    'ǹ': 'n', 'Ǹ': 'N',
    'ṯ': 't', 'Ṯ': 'T',
    'ḳ': 'k', 'Ḳ': 'K',
    'ṉ': 'n', 'Ṉ': 'N',
    'ṗ': 'p', 'Ṗ': 'P',
    'ẖ': 'h', 'Ẕ': 'Z',
}
# Normalization function (removes extra spaces)
def normalizeString(s):
    return s.strip()

def readLangs(file_path, lang1, lang2, reverse=False):
    """Reads tab-separated English-Nepali pairs from a file."""
    print("Reading lines...")
    with open(file_path, encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    pairs = [list(map(normalizeString, line.split("\t"))) for line in lines if "\t" in line]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def filterPair(p):
    """Filters sentence pairs based on max length and empty sentences."""
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH and p[0] != '' and p[1] != ''

def filterPairs(pairs):
    """Applies filtering to all pairs."""
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(file_path, lang1, lang2, reverse=False):
    """Loads, cleans, and processes data."""
    input_lang, output_lang, pairs = readLangs(file_path, lang1, lang2, reverse)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(f"{input_lang.name}: {input_lang.n_words}")
    print(f"{output_lang.name}: {output_lang.n_words}")
    return input_lang, output_lang, pairs

def simplify_special_characters(text):

    text = re.sub(r'[^\w\s]', '', text)  # Removes nachahine symbols which may give error

    simplified_text = ''.join(special_character_mapping.get(char, char) for char in text)
    return simplified_text

romanized_text = transliterate(devanagari_text, sanscript.DEVANAGARI, sanscript.ITRANS)
romanized_text=romanized_text.lower()
#
# File path to your dataset
file_path = "data/eng-nep.txt"

# Process data
input_lang, output_lang, pairs = prepareData(file_path, "eng", "nep", True)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData(file_path, "eng", "nep", True)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    # train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
    #                            torch.LongTensor(target_ids).to(device))

    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

hidden_size = 128
batch_size = 32

input_lang_nep, output_lang_eng_forne = get_dataloader(batch_size)

encoderr = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoderr = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# Load the checkpoint
checkpoint = torch.load(r'saved_models/translation_models/PTnep2engV2.pt', map_location=device)

# Load the state dictionaries
encoderr.load_state_dict(checkpoint['encoder_state_dict'])
decoderr.load_state_dict(checkpoint['decoder_state_dict'])

print("Model loaded successfully!")

def talkne2en(input_sentence):
    with torch.no_grad():
        output_words, attentions = evaluate(encoderr, decoderr, input_sentence, input_lang_nep, output_lang_eng_forne)
        print('input =', input_sentence)
        print('output =', " ".join(output_words))
    # return (output_words)
    return " ".join(output_words)


# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 




SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang2, lang1, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang2, lang1, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# input_lang, output_lang, pairs = prepareData('eng', 'nep', True)
# for pair in pairs:
#     print(pair)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) 
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None 

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) 
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() 

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'nep', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words



hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
encoder.eval()
decoder.eval()

def yap(input_sentence):
    with torch.no_grad():
        output_words = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        return ' '.join(output_words)
        
 # Load the previously trained model
device = torch.device("cpu")  # or "cuda"  <<cpu better i say>>
checkpoint = torch.load(r'saved_models/translation_models/PTeng2fraV3.pt', map_location=device)

encoder = checkpoint['encoder_class'](checkpoint['input_lang_n_words'], checkpoint['hidden_size']).to(device)
decoder = checkpoint['decoder_class'](checkpoint['hidden_size'], checkpoint['output_lang_n_words']).to(device)
attention = checkpoint['attention_class'](checkpoint['hidden_size'])

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

print("Model loaded successfully!")       


# for french translation


START_TOKEN = 0
END_TOKEN = 1

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "START", 1: "END"}
        self.n_words = 2  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLanguages(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLanguages(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(START_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) 
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None 

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = AttentionMechanism(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(START_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) 
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() 

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def indexesFromSentence(language, sentence):
    return [language.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(language, sentence):
    indexes = indexesFromSentence(language, sentence)
    indexes.append(END_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(END_TOKEN)
        tgt_ids.append(END_TOKEN)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == END_TOKEN:
                decoded_words.append('<END>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = RNNEncoder(input_lang.n_words, hidden_size).to(device)
decoder = AttentionDecoder(hidden_size, output_lang.n_words).to(device)
encoder.eval()
decoder.eval()

def talktuah(input_sentence):
    with torch.no_grad():

        input_sentence = input_sentence.lower()
        output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
        output_sentence = ' '.join(output_words)  
        return output_sentence
        

# Load the checkpoint
device = torch.device("cpu")  
checkpoint = torch.load(r'saved_models/translation_models/PTeng2fraV3.pt', map_location=device)

encoder = checkpoint['encoder_class'](checkpoint['input_lang_n_words'], checkpoint['hidden_size']).to(device)
decoder = checkpoint['decoder_class'](checkpoint['hidden_size'], checkpoint['output_lang_n_words']).to(device)
attention = checkpoint['attention_class'](checkpoint['hidden_size'])

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

print("Model loaded successfully!asdassssssssssssssssssssssssssssssss")
print(talktuah("Soyez gentils !"))

# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 


# Load any image format using PIL and convert to OpenCV format
def load_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")  # Convert to RGB
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

# Function to get the best text with the highest probability
def get_best_text(results):
    if not results:
        return None, 0  # No text found
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence (highest first)
    
    # Combine detected words into a full sentence while preserving order
    sorted_by_position = sorted(results, key=lambda x: (x[0][1][1], x[0][0][0]))  # Sort by y, then x position
    sentence = " ".join([res[1] for res in sorted_by_position])
    
    return sentence, results[0][2]  # Return the full text and highest confidence




# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # 








app = Flask(__name__)
CORS(app)
sentence = "टम धेरै सुन्दर छ"
print(talkne2en(sentence))
print(talktuah("Soyez gentils !"))
UPLOAD_FOLDER = "uploads"
EMBEDDINGS_FOLDER = "embeddings"
AUTOSPEECH_FOLDER="autospeech"
IMAGE_FOLDER="ImageToText"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUTOSPEECH_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)  # Create folder for embeddings
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Initialize UI and Toolbox conditionally
ui = UI()  # Initialize UI for both production and development

synthesizer = None  # Will be initialized when needed
speaker_name = "user01"  # Default speaker name

# Load models dynamically when needed
def load_models():
    global synthesizer
    if synthesizer is None:
        synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))  # Updated path
    if not encoderv.is_loaded():
        encoderv.load_model(Path("saved_models/default/encoder.pt"))  # Updated path
    if not vocoder.is_loaded():
        vocoder.load_model(Path("saved_models/default/vocoder.pt"))  # Updated path

def record_one(file_path, sample_rate=16000):
    """Loads and resamples a .wav file for processing."""
    if not os.path.exists(file_path):
        return None, "File not found"

    # Load the audio with librosa (it automatically resamples)
    wav, sr = librosa.load(file_path, sr=sample_rate)

    if sr != sample_rate:
        print(f"Warning: File sample rate ({sr}Hz) does not match expected {sample_rate}Hz")

    return wav, None

def add_real_utterance(wav, name, speaker_name):
    """Processes a recorded utterance by generating spectrogram and embeddings."""
    spec = Synthesizer.make_spectrogram(wav)

    if ui:  # Only call ui methods if ui is initialized
        ui.draw_spec(spec, "current")

    if not encoderv.is_loaded():
        encoderv.load_model(Path("saved_models/default/encoder.pt"))

    encoder_wav = encoderv.preprocess_wav(wav)
    embed, partial_embeds, _ = encoderv.embed_utterance(encoder_wav, return_partials=True)

    utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)

    if ui:  # Only register and draw if ui is initialized
        ui.register_utterance(utterance)
        ui.draw_embed(embed, name, "current")

    return utterance

@app.route('/upload', methods=['POST'])
def upload_audio():
    print("Received request:", request.files)  # Debugging line
    """Handles file upload and processes it through record functions."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if file.filename.endswith('.wav'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"Saved file at: {file_path}")
        # Read the file
        wav, error = record_one(file_path)
        if error:
            return jsonify({"error": error}), 400

        # Process the utterance
        name = f"{speaker_name}_rec_{np.random.randint(100000)}"
        utterance = add_real_utterance(wav, name, speaker_name)

        # Save the embedding to a file
        embedding_file_path = os.path.join(EMBEDDINGS_FOLDER, f"embedding.npy")
        np.save(embedding_file_path, utterance.embed)  # Save the embedding as a .npy file

        return jsonify({
            "message": "File processed successfully",
            "filename": file.filename,
            "path": file_path,
            "spectrogram_shape": utterance.spec.shape if utterance else "Error",
            "embedding_file": embedding_file_path  # Provide path to saved embedding file
        })
    else:
        return jsonify({"error": "Invalid file type, only .wav allowed"}), 400

from flask import send_file

@app.route('/synthesize_and_vocode', methods=['POST'])
def synthesize_and_vocode():
    """Handles text synthesis and vocoding in one request."""
    data = request.json
    text = data.get("text")
    embedding_file = "embeddings/embedding.npy"

    if not text or not embedding_file:
        return jsonify({"error": "Text and embedding_file are required"}), 400

    if ui is None:
        return jsonify({"error": "UI is not initialized"}), 500

    try:
        embedding = np.load(embedding_file)
    except Exception as e:
        return jsonify({"error": f"Error loading embedding file: {str(e)}"}), 400

    embedding_name = os.path.basename(embedding_file).split('.')[0]
    utterance = Utterance(embedding_name, speaker_name, None, None, embedding, None, False)

    load_models()
    ui.log("Generating the mel spectrogram...")
    
    texts = text.split("\n")
    embeds = [utterance.embed] * len(texts)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = np.concatenate(specs, axis=1)

    ui.log("Generating waveform...")
    wav = vocoder.infer_waveform(spec)

    wav = wav / np.abs(wav).max() * 0.97  # Normalize

    # Save the file to disk
    wav_path = os.path.join(UPLOAD_FOLDER, "synthesized_audio.wav")
    sf.write(wav_path, wav, Synthesizer.sample_rate)

    return send_file(
        wav_path,
        mimetype="audio/wav",
        as_attachment=True,
        download_name="audio.wav"
    )

      
# output_audio_url
@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    user_input = data.get('input', '')
    source_language = data.get('source_language', '')
    target_language = data.get('target_language', '')

    print(f"User input: {user_input}")
    print(f"Source language: {source_language}")
    print(f"Target language: {target_language}")

    try:
        # Choose the appropriate translation function based on the source language
        if source_language == "nep" and target_language == "eng":
            result = talkne2en(user_input)  # Nepali to English
        elif source_language == "fra" and target_language == "eng":
            # Remove special characters from user input
            cleaned_input = re.sub(r'[^\w\s]', ' ', user_input)
            result = talktuah(cleaned_input)  # French to English
        else:
            raise ValueError("Unsupported source or target language pair")

        print(result)
        op_cleaned = result.replace("<EOS>", "").strip()
        op = op_cleaned
        print(f"Translation result: {op}")

        if op is None:
            op = "Error: No translation result generated."

        print(op)
        ops_cleaned = op.replace("<END>", "").strip()
        ops = ops_cleaned
        print(f"Translation result: {ops}")

        if ops is None:
            ops = "Error: No translation result generated."

    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

    response = jsonify({
        "translatedText": ops
    })
    print(f"Response sent to frontend: {response.get_json()}")  # Log the response being sent
    return response

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return {"error": "Text is required"}, 400

    engine = pyttsx3.init()
    output_path = "output.wav"
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    print("response sent")
    return send_file(output_path, mimetype="audio/wav", as_attachment=True)

@app.route('/autotext', methods=['POST'])
def autotext():
    data = request.get_json()
    text = data.get("text", "")
    def get_language_full_name(text):
        lang_code = detect(text)
        language = pycountry.languages.get(alpha_2=lang_code)
        if language:
            return language.name.split(" (")[0]  # Remove extra details like " (macrolanguage)"
        return "Unknown Language"

    # text = "I am working hard right now."
    detected_language = get_language_full_name(text)
    print(detected_language)  
    return jsonify({"detected language": detected_language})

@app.route('/autospeech',methods=['POST'])
def autospeech():

    # Define constants for recording
    SAMPLING_RATE = 16000  # 16 kHz sampling rate
    CHANNELS = 1  # Mono audio
    FORMAT = pyaudio.paInt16  # 16-bit audio format
    CHUNK_SIZE = 1024  # Number of frames per buffer
    RECORD_DURATION = 5  # Duration in seconds

    # Language code to full name mapping
    LANGUAGE_MAP = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "hi": "Hindi",
        "ne": "Nepali",
        "ar": "Arabic",
        "pt": "Portuguese",
        "bn": "Bengali",
        "ur": "Urdu",
        "tr": "Turkish",
        "vi": "Vietnamese"
    }

    def detect_language_and_transcribe(file_path, model_size="medium", device="cpu", compute_type="int8"):
        # Initialize the WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Perform transcription
        segments, info = model.transcribe(file_path, beam_size=5, language=None)  # Auto language detection
        transcribed_text = " ".join([segment.text for segment in segments if hasattr(segment, 'text')])

        # Detect language
        detected_lang_code = info.language
        detected_lang_name = LANGUAGE_MAP.get(detected_lang_code, "Unknown Language")

        # Print detected language and transcription
        print(f"Detected language: '{detected_lang_name}' ({detected_lang_code}) with probability {info.language_probability:.6f}")
        print(f"Transcribed Text: {transcribed_text}")
        return detected_lang_name, transcribed_text

    print("Received request:", request.files)  # Debugging line
    # "Handles file upload and processes it through record functions."
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if file.filename.endswith('.wav'):
        file_path = os.path.join(AUTOSPEECH_FOLDER, file.filename)
        file.save(file_path)
        print(f"Saved file at: {file_path}")
        # Read the file
        wav, error = record_one(file_path)
        if error:
            return jsonify({"error": error}), 400
        
    # audio_file = record_audio(duration=5, file_path="recorded_audio.wav")

    # Detect language and transcribe
    detected_language, transcribed_text = detect_language_and_transcribe(file_path)
    print(f"Detected Language: {detected_language}")
    print(f"Transcribed Text: {transcribed_text}")

    return jsonify({
        "detected_language": detected_language,
        "transcribed_text": transcribed_text
    })


@app.route('/image', methods=['POST'])
def image():
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    image = request.files['image']

    if image.filename == '':
        return {"error": "No selected file"}, 400

    image_path = os.path.join(IMAGE_FOLDER, image.filename)
    image.save(image_path)  # Save the image

    # Load the image
    image = load_image(image_path)

    # Initialize OCR readers for Chinese, Nepali, French, and English
    reader_chinese = easyocr.Reader(['ch_tra', 'en'])  # Chinese OCR
    reader_nepali = easyocr.Reader(['hi', 'mr', 'ne', 'en'])  # Nepali OCR
    reader_french = easyocr.Reader(['fr', 'en'])  # French OCR
    reader_english = easyocr.Reader(['en'])  # English OCR

    # Step 1: Run OCR on all four languages
    text_chinese_results = reader_chinese.readtext(image)
    text_nepali_results = reader_nepali.readtext(image)
    text_french_results = reader_french.readtext(image)
    text_english_results = reader_english.readtext(image)

    # Get best results for each language
    best_chinese, conf_chinese = get_best_text(text_chinese_results)
    best_nepali, conf_nepali = get_best_text(text_nepali_results)
    best_french, conf_french = get_best_text(text_french_results)
    best_english, conf_english = get_best_text(text_english_results)

    # Determine the best result overall
    best_text = None
    best_confidence = 0
    best_language = None

    if best_chinese and conf_chinese > best_confidence:
        best_text, best_confidence, best_language = best_chinese, conf_chinese, "Chinese"

    if best_nepali and conf_nepali > best_confidence:
        best_text, best_confidence, best_language = best_nepali, conf_nepali, "Nepali"

    if best_french and conf_french > best_confidence:
        best_text, best_confidence, best_language = best_french, conf_french, "French"

    if best_english and conf_english > best_confidence:
        best_text, best_confidence, best_language = best_english, conf_english, "English"

    if best_text:
        response = {
            "text": best_text,
            "language": best_language         
        }
    else:
        response = {"message": "Extracted Text: Not Available", "text": None}
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# curl -X POST http://127.0.0.1:5000/upload ^ -F "file=@C:/Users/Acer/Desktop/records/4.wav"
# curl -X POST http://127.0.0.1:5000/synthesize_and_vocode ^ -H "Content-Type: application/json" ^  -d "{\"text\": \"Hello, how are you?\", \"embedding_file\": \"embeddings/embedding.npy\"}" 
# curl -X POST http://localhost:5000/submit ^-H "Content-Type: application/json" ^-d "{\"input\": \"c est un gentleman\", \"source_language\": \"fra\", \"target_language\": \"eng\"}"
# curl -X POST http://localhost:5000/submit ^-H "Content-Type: application/json" ^-d "{\"input\": \"तपाईं धेरै सुन्दर हुनुहुन्छ\", \"source_language\": \"nep\", \"target_language\": \"eng\"}"
# curl -X POST http://127.0.0.1:5000/autotext -H "Content-Type: application/json" -d "{\"text\": \"File dans ta voiture !\"}"
# curl -X POST http://127.0.0.1:5000/autospeech -H "Content-Type: multipart/form-data" -F "file=@C:\Users\Acer\Desktop\2Real-Time-Voice-Cloning\records\nepali.wav"
# curl -X POST "http://127.0.0.1:5000/tts" -H "Content-Type: application/json" -d "{\"text\": \"Hello, this is a test.\"}" --output output.wav
# curl -X POST -F "image=@\"C:\Users\Acer\Desktop\NEPALI.png\"" http://127.0.0.1:5000/image
# ngrok http 5000