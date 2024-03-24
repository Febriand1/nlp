# -*- coding: utf-8 -*-
import json
import os
import pickle

import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

# Membuat sesi TensorFlow dengan opsi log_device_placement untuk mencatat perangkat yang digunakan oleh operasi.
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Mendefinisikan path output untuk menyimpan hasil-hasil pelatihan
path = "output_dir/"
# Mencoba membuat direktori output, jika gagal (kemungkinan karena sudah ada), akan dilanjutkan.
try:
    os.makedirs(path)
except:
    pass

# Membaca dataset dari file teks CSV dengan delimiter "|" dan tanpa header.
dataset = pd.read_csv('./dataset/clean_dataset2.txt', delimiter="|", header=None,lineterminator='\n')

# Mengambil sebagian data untuk data validasi dan menyimpannya dalam file CSV.
dataset_val = dataset.iloc[100:].to_csv('output_dir/val1.csv')

# Mengambil sebagian data untuk data pelatihan.
dataset_train = dataset.iloc[:100]

# Mengambil pertanyaan dari data pelatihan.
questions_train = dataset_train.iloc[:, 0].values.tolist()

# Mengambil jawaban dari data pelatihan.
answers_train = dataset_train.iloc[:, 1].values.tolist()

# Mengambil pertanyaan dari data pelatihan untuk pengujian.
questions_test = dataset_train.iloc[:, 0].values.tolist()

# Mengambil jawaban dari data pelatihan untuk pengujian.
answers_test = dataset_train.iloc[:, 1].values.tolist()

# Mendefinisikan fungsi untuk menyimpan tokenizer.
def save_tokenizer(tokenizer):
    with open('output_dir/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Mendefinisikan fungsi untuk menyimpan konfigurasi.
def save_config(key, value):
    data = {}
    if os.path.exists(path + 'config.json'):
        with open(path + 'config.json') as json_file:
            data = json.load(json_file)

    data[key] = value
    with open(path + 'config.json', 'w') as outfile:
        json.dump(data, outfile)

# Mendefinisikan pola untuk karakter target yang akan dihilangkan dari teks.
target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'

# Membuat objek tokenizer dengan filter karakter yang telah ditentukan.
tokenizer = Tokenizer(filters=target_regex, lower=True)

# Melatih tokenizer pada teks pertanyaan dan jawaban.
tokenizer.fit_on_texts(questions_train + answers_train + questions_test + answers_test)

#  Menyimpan tokenizer ke dalam file pickle.
save_tokenizer(tokenizer)

# Menghitung ukuran kosakata berdasarkan jumlah kata yang ditemukan oleh tokenizer.
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Menyimpan ukuran kosakata ke dalam konfigurasi.
save_config('VOCAB_SIZE', VOCAB_SIZE)
print('Vocabulary size : {}'.format(VOCAB_SIZE))

# Melakukan tokenisasi pada teks pertanyaan untuk data pelatihan.
tokenized_questions_train = tokenizer.texts_to_sequences(questions_train)

# Menghitung panjang maksimum pertanyaan untuk data pelatihan.
maxlen_questions_train = max([len(x) for x in tokenized_questions_train])
save_config('maxlen_questions', maxlen_questions_train)

# Membuat data input terenkripsi (encoder) untuk data pelatihan.
encoder_input_data_train = pad_sequences(tokenized_questions_train, maxlen=maxlen_questions_train, padding='post')

# Melakukan tokenisasi pada teks pertanyaan untuk data pengujian.
tokenized_questions_test = tokenizer.texts_to_sequences(questions_test)

# Menghitung panjang maksimum pertanyaan untuk data pengujian.
maxlen_questions_test = max([len(x) for x in tokenized_questions_test])
save_config('maxlen_questions', maxlen_questions_test)

# Membuat data input terenkripsi (encoder) untuk data pengujian.
encoder_input_data_test = pad_sequences(tokenized_questions_test, maxlen=maxlen_questions_test, padding='post')

# Melakukan tokenisasi pada teks jawaban untuk data pelatihan.
tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)

# Menghitung panjang maksimum jawaban untuk data pelatihan.
maxlen_answers_train = max([len(x) for x in tokenized_answers_train])
save_config('maxlen_answers', maxlen_answers_train)

# Membuat data input terenkripsi (decoder) untuk data pelatihan.
decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')

# Melakukan tokenisasi pada teks jawaban untuk data pengujian.
tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)

# Menghitung panjang maksimum jawaban untuk data pengujian.
maxlen_answers_test = max([len(x) for x in tokenized_answers_test])
save_config('maxlen_answers', maxlen_answers_test)

# Membuat data input terenkripsi (decoder) untuk data pengujian.
decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')

# Mempersiapkan data output terenkripsi (decoder) untuk data pelatihan.
for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:]
    
# Melakukan padding pada data jawaban untuk data pelatihan.    
padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')

# Membuat data output terenkripsi (decoder) untuk data pelatihan.
decoder_output_data_train = to_categorical(padded_answers_train, num_classes=VOCAB_SIZE)

# Mempersiapkan data output terenkripsi (decoder) untuk data pengujian.
for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]
    
# Melakukan padding pada data jawaban untuk data pengujian.
padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')

# Membuat data output terenkripsi (decoder) untuk data pengujian.
decoder_output_data_test = to_categorical(padded_answers_test, num_classes=VOCAB_SIZE)

# Mendefinisikan input untuk encoder.
enc_inp = Input(shape=(None,))

# Membuat layer embedding untuk encoder.
enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inp)

# Membuat layer LSTM yang menggunakan dua arah (bidirectional) untuk encoder.
enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)

# Menggabungkan state_h dari kedua arah LSTM untuk encoder.
state_h = Concatenate()([forward_h, backward_h])

# Menggabungkan state_c dari kedua arah LSTM untuk encoder.
state_c = Concatenate()([forward_c, backward_c])

# Membuat tuple yang berisi state_h dan state_c untuk encoder.
enc_states = [state_h, state_c]

# Mendefinisikan input untuk decoder.
dec_inp = Input(shape=(None,))

# Membuat layer embedding untuk decoder.
dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inp)

# Membuat layer LSTM untuk decoder.
dec_lstm = LSTM(256 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)

# Meneruskan input dan state encoder ke layer decoder LSTM.
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

# Membuat layer Dense dengan fungsi aktivasi softmax untuk output decoder.
dec_dense = Dense(VOCAB_SIZE, activation=softmax)

# Meneruskan output dari layer decoder LSTM ke layer Dense.
output = dec_dense(dec_outputs)

# Mendefinisikan path untuk log TensorBoard.
logdir = os.path.join(path, "logs")

# Membuat callback TensorBoard untuk melihat log pelatihan.
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Membuat callback ModelCheckpoint untuk menyimpan model selama pelatihan.
checkpoint = ModelCheckpoint(os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='auto', period=100)

# Membuat model dengan input dan output yang telah didefinisikan.
model = Model([enc_inp, dec_inp], output)

# Mengkompilasi model dengan pengoptimal RMSprop dan fungsi kerugian categorical_crossentropy.
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

# Menampilkan ringkasan model.
model.summary()

#  Mendefinisikan ukuran batch.
batch_size = 10

# Mendefinisikan jumlah epoch.
epochs = 10

# Melatih model dengan data pelatihan dan validasi, menggunakan callback yang telah didefinisikan.
model.fit([encoder_input_data_train, decoder_input_data_train],
          decoder_output_data_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
          callbacks=[tensorboard_callback, checkpoint])

# Menyimpan model setelah pelatihan.
model.save(os.path.join(path, 'model-' + path.replace("/", "") + '.h5'))
