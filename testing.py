from iteung import reply
# import pandas as pd

# data = reply.get_val_data()
# val_q = []
# val_a = []
# bot_a = []
# for i, v in data.iterrows():
#     val_question = v['0']
#     val_answer = v['1'].replace('<START>', '').replace('<END>', '').strip()
#     val_q.append(val_question)
#     val_a.append(val_answer)
#     bot_a.append(reply.botReply(val_question)[0])

# bot_test_dataframe = pd.DataFrame(
#     {
#         'question': val_q,
#         'answer': val_a,
#         'bot': bot_a
#     }
# )

# reply.bot_test_dataframe.to_csv('output_dir/test_result.csv', index=False)

while True:
    message = input("Kamu: ")
    return_message, status = reply.botReply(message)
    print(f"ITeung: {return_message}")

# import pandas as pd
# from iteung import reply

# # Daftar pertanyaan
# listPertanyaan = [
#     "Hai",
#     "Apa kabar?",
#     "Siapa nama kamu?",
#     "kamu sehat",
#     "kamu suka makan apa?",
#     "udah makan?",
#     "kamu suka makanan apa?",
#     "kamu suka minuman apa?",
# ]

# # Inisialisasi list jawaban dan akurasi
# listJawaban = []
# listAkurasi = []

# # Mendapatkan jawaban dan akurasi untuk setiap pertanyaan
# for message in listPertanyaan:
#     return_message, status , dec_outputs, akurasi= reply.botReply(message)
#     listJawaban.append(return_message)
#     listAkurasi.append(akurasi)

# # Membuat DataFrame menggunakan pandas
# df = pd.DataFrame({
#     'Pertanyaan': listPertanyaan,
#     'Jawaban': listJawaban,
#     'Akurasi': listAkurasi
# })

# # Menyimpan DataFrame ke dalam file Excel
# df.to_excel('hasil_bot.xlsx', index=False)


# from iteung import reply
# import pandas as pd

# data = reply.get_val_data()
# val_q = []
# val_a = []
# bot_a = []
# for i, v in data.iterrows():
#     val_question = v['0']
#     val_answer = v['1'].replace('<START>', '').replace('<END>', '').strip()
#     val_q.append(val_question)
#     val_a.append(val_answer)
#     return_message, _,_,kecocokan = reply.botReply(val_question)
#     bot_a.append(return_message)

# bot_test_dataframe = pd.DataFrame(
#     {
#         'question': val_q,
#         'answer': val_a,
#         'bot': bot_a,
#         'kecocokan':kecocokan
#     }
# )

# bot_test_dataframe.to_excel('test_result.xlsx', index=False)

