import telebot
import sqlite3
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

conn = sqlite3.connect('/content/breeds.db', check_same_thread=False)
cursor = conn.cursor()

with open('/content/content/models/model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)


model.load_weights('/content/content/models/model.h5')

bot = telebot.TeleBot("7064253940:AAG27j_opz7ltR3iktOuiywhQ-oHDtuFx3c")  # Замените "YOUR_API_TOKEN" на фактический API-токен вашего бота

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Пришли мне фотографию собаки, и я определю её породу.")

@bot.message_handler(func=lambda message: True)
def handle_non_photo(message):
    if 'photo' not in message.content_type:
        bot.reply_to(message, "Вы не отправили фотографию")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo_id = message.photo[-1].file_id
    file_info = bot.get_file(photo_id)
    downloaded_file = bot.download_file(file_info.file_path)

    photo_path = 'photo.jpg'
    with open(photo_path, 'wb') as f:
        f.write(downloaded_file)


    img = image.load_img(photo_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    predictions = model.predict(img)
    breed_index = np.argmax(predictions)
    print(type(breed_index))
    breed_index = int(breed_index)
    cursor.execute("SELECT name, description FROM breeds WHERE breed_index = ?", (breed_index,))
    result = cursor.fetchone()


    if result:
        breed_name, description = result
        bot.reply_to(message, f"Порода: {breed_name}\n\nОписание: {description}")
    else:
        bot.reply_to(message, "К сожалению, не удалось определить породу собаки.")


bot.polling()
