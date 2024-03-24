from iteung import reply
import pandas as pd
import os

file_path = "hasil_akurasi_dataset.xlsx"
adaFile = False

listJawaban = []hai
listAkurasi = []
listPertanyaan = []

if os.path.exists(file_path):
    print("File Excel ada.")
    try:
        # Membaca data dari file Excel
        data = pd.read_excel(file_path)
        # Mengambil data dari kolom yang ditentukan
        listJawaban = data["Jawaban"].tolist()
        listAkurasi = data["Akurasi"].tolist()
        listPertanyaan = data["Pertanyaan"].tolist()
        adaFile = True
    except pd.errors.EmptyDataError:
        print("File Excel Kosong.")
else:
    print("File Excel tidak ada.")

while True:
    message = input("Kamu: ")
    if message == "exit":

        break
    return_message, status , dec_outputs, akurasi= reply.botReply(message)
    listJawaban.append(return_message)
    listAkurasi.append(akurasi)
    listPertanyaan.append(message)

    print(f"ITeung: {return_message}")

    df = pd.DataFrame({
        'Pertanyaan': listPertanyaan,
        'Jawaban': listJawaban,
        'Akurasi': listAkurasi
    })

    if adaFile:
        try:
            df.to_excel(file_path, index=False)
        except PermissionError:
            print("File Excel sedang dibuka. Tutup file tersebut dan coba lagi.")

    else:
        df.to_excel(file_path, index=False)
