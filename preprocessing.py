# -*- coding: utf-8 -*-
"""ITeung

# Preprocessing
"""

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import io
import os
import re
import requests
import csv
import datetime
import numpy as np
import pandas as pd
import random
import pickle

# Membuat objek stemmer dari bahasa Indonesia menggunakan Sastrawi.
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Membuat pola regex untuk mencocokkan tanda baca.
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

# Daftar kata yang menunjukkan ketidakfahaman.
unknowns = ["gak paham","kurang ngerti","I don't know"]

# Membaca file CSV yang berisi daftar kata slang dalam bahasa Indonesia ke dalam bentuk numpy array.
list_indonesia_slang = pd.read_csv('./dataset/daftar-slang-bahasa-indonesia.csv', header=None).to_numpy()

# Membuat kamus kosong untuk menyimpan kata slang dan padanan aslinya.
data_slang = {}

# Mengisi kamus data_slang dengan pasangan kata slang dan kata aslinya.
for key, value in list_indonesia_slang:
    data_slang[key] = value

# Fungsi yang mengembalikan nilai dari kamus berdasarkan kunci yang diberikan.
def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)

# Fungsi untuk memeriksa apakah kata merupakan kata slang atau bukan, dan mengembalikan padanan aslinya jika kata tersebut adalah kata slang.
def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)
    if slang_result:
        return slang_result
    return word_input

# Fungsi untuk normalisasi kalimat, termasuk penghapusan tanda baca, stemming, dan mengganti kata slang dengan kata aslinya.
def normalize_sentence(sentence):
  sentence = punct_re_escape.sub('', sentence.lower())
  sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','')
  sentence = sentence.replace('teung', '')
  sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
  sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
  sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
  sentence = ' '.join(sentence.split())
  if sentence:
    sentence = sentence.strip().split(" ")
    normal_sentence = " "
    for word in sentence:
      normalize_word = check_normal_word(word)
      root_sentence = stemmer.stem(normalize_word)
      normal_sentence += root_sentence+" "
    return punct_re_escape.sub('',normal_sentence)
  return sentence

# Membaca file CSV yang berisi data pertanyaan dan jawaban ke dalam pandas DataFrame.
df = pd.read_csv('./dataset/dataset.csv', sep='|',usecols= ['question','answer'])
df.head()

print('Membuat kamus kosong untuk menghitung panjang pertanyaan dan jawaban.')
# question_length = {}
# answer_length = {}

# # Melakukan iterasi pada setiap baris dalam DataFrame.
# for index, row in df.iterrows():
#   # Normalisasi pertanyaan.
#   question = normalize_sentence(row['question'])
#   question = normalize_sentence(question)
#   question = stemmer.stem(question)
  
#   # Memeriksa apakah panjang pertanyaan sudah ada dalam kamus.
#   if question_length.get(len(question.split())):
#     question_length[len(question.split())] += 1
#   else:
#     question_length[len(question.split())] = 1
    
#   # Memeriksa apakah panjang jawaban sudah ada dalam kamus.  
#   if answer_length.get(len(str(row['answer']).split())):
#     answer_length[len(str(row['answer']).split())] += 1
#   else:
#     answer_length[len(str(row['answer']).split())] = 1

# question_length

# answer_length

# # Mengambil nilai dan kunci dari kamus panjang pertanyaan.
# val_question_length = list(question_length.values())
# key_question_length = list(question_length.keys())

# # Menggabungkan kunci dan nilai dari kamus pertanyaan menjadi pasangan kunci-nilai.
# key_val_question_length = list(zip(key_question_length, val_question_length))

# # Membuat DataFrame dari pasangan kunci-nilai panjang pertanyaan.
# df_question_length = pd.DataFrame(key_val_question_length, columns=['length_data', 'total_sentences'])
# df_question_length.sort_values(by=['length_data'], inplace=True)
# df_question_length.describe()

# # Mengambil nilai dan kunci dari kamus panjang jawaban.
# val_answer_length = list(answer_length.values())
# key_answer_length = list(answer_length.keys())

# # Menggabungkan kunci dan nilai dari kamus jawaban menjadi pasangan kunci-nilai.
# key_val_answer_length = list(zip(key_answer_length, val_answer_length))

# # Membuat DataFrame dari pasangan kunci-nilai panjang jawaban.
# df_answer_length = pd.DataFrame(key_val_answer_length, columns=['length_data', 'total_sentences'])
# df_answer_length.sort_values(by=['length_data'], inplace=True)
# df_answer_length.describe()

# # Variabel yang tidak digunakan.
# data_length = 0
    
# Menentukan nama file untuk menyimpan data yang telah diproses.
#filename = open('./dataset/clean_qa.txt', 'a+')
print("Menentukan nama file untuk menyimpan data yang telah diproses.")
filename= './dataset/clean_dataset2.txt'

# Membuka file untuk menulis data.
print("Membuka file untuk menulis data.")
with open(filename, 'w', encoding='utf-8') as f:
  print('Melakukan iterasi pada setiap baris dalam DataFrame.')  
  for index, row in df.iterrows():
    question = normalize_sentence(str(row['question']))
    question = normalize_sentence(question)
    question = stemmer.stem(question)

    answer = str(row['answer']).lower().replace('iteung', 'aku').replace('\n', ' ')
    
    print('\n')
    print(answer)
    print('\n')
    
    print('Memeriksa apakah panjang pertanyaan dan jawaban memenuhi kriteria yang ditentukan.')
    if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
        
      print('Membuat string yang akan ditulis ke file.')  
      body="{"+question+"}|<START> {"+answer+"} <END>"
      
      print('Menulis string ke dalam file.')
      print(body, file=f)
      #filename.write(f"{question}\t<START> {answer} <END>\n")
      
print('beres')
#filename.close()
