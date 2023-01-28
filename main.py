import random
import json
import pickle
import numpy as np
import nltk
import sqlite3
import os
import datetime
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import warnings

from kivy.clock import Clock
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.label import MDLabel
from kivy.properties import StringProperty, NumericProperty
from kivy.core.text import LabelBase
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

Window.size = (350, 650)

# testing branch

# KivyMD
class Command(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty()
    font_name = "Poppins"
    font_size = 17


class Response(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty()
    font_name = "Poppins"
    font_size = 17


# ML
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('JSON_FILES/intents.json').read())

words = pickle.load(open('pickle_files/words.pkl', 'rb'))
classes = pickle.load(open('pickle_files/classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')
phq_total_score = 0
gad_total_score = 0


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


class OkBot(MDApp):
    dialog = None
    dialogs = None
    dialog_login = None

    if os.path.exists("okBot.db"):
        conn = sqlite3.connect("okBot.db")
        c = conn.cursor()
    else:
        conn = sqlite3.connect("okBot.db")
        c = conn.cursor()

        query = (''' CREATE TABLE IF NOT EXISTS accounts 
                    (
                    uname VARCHAR(50) NOT NULL PRIMARY KEY,
                    pwd VARCHAR(20) NOT NULL
                    );
                ''')
        c.execute(query)
        query2 = (''' CREATE TABLE IF NOT EXISTS CHAT
                    (
                    INPUT VARCHAR(50) NOT NULL,
                    OUTPUT VARCHAR(50) NOT NULL,
                    USERNAME VARCHAR(50) NOT NULL, FOREIGN KEY(USERNAME) REFERENCES accounts(uname)
                    );
                    ''')
        c.execute(query2)

    def change_screen(self, name):
        screen_manager.current = name

    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file("Kivy Files/main.kv"))
        screen_manager.add_widget(Builder.load_file("Kivy Files/Chats.kv"))
        screen_manager.add_widget(Builder.load_file("Kivy Files/phq.kv"))
        screen_manager.add_widget(Builder.load_file("Kivy Files/gad.kv"))
        screen_manager.add_widget(Builder.load_file("Kivy Files/about.kv"))
        screen_manager.add_widget(Builder.load_file("Kivy Files/helpline.kv"))
        return screen_manager

    def main(self):
        screen_manager.current = "main"


    def bot_name(self):
        screen_manager.current = "chats"

    def login(self, bot_name, password):
        self.c.execute("SELECT * FROM accounts")
        username = []
        for i in self.c.fetchall():
            username.append(i[0])
        if bot_name.text in username and bot_name.text != "":
            self.c.execute(f"select pwd from accounts where uname='{bot_name.text}'")
            for j in self.c:
                if password.text == j[0]:
                    print("Logged in!")
                    screen_manager.current = "chats"
                else:
                    self.invalidDialog()
                    screen_manager.current = "main"

    def invalidDialog(self):
        if not self.dialog_login:
            self.dialog_login = MDDialog(
                text="Bukti kelayakan tidak tepat",
                size_hint=[0.5, None],
                pos_hint={"center_x": 0.5, "center_y": 0.1},
                buttons=[
                    MDFlatButton(
                        text="Cuba Lagi",
                        on_release=self.close_dialog_login
                    ),
                ],
            )
        self.dialog_login.open()

    def register(self, bot_name, password):
        try:
            self.c.execute(f"insert into accounts values('{bot_name.text}','{password.text}')")
            self.conn.commit()
            screen_manager.current = "chats"
        except:
            screen_manager.current = "main"

    def phq(self):
        screen_manager.current = "phq"

    def gad(self):
        screen_manager.current = "gad"

    def about(self):
        screen_manager.current = "about"

    def helpline(self):
        screen_manager.current = "helpline"

    def response(self, *args):
        ints = predict_class(value)
        response = get_response(ints, intents)
        screen_manager.get_screen('chats').chat_list.add_widget(Response(text=response, size_hint_x=.75))

    def send(self, bot_name, text_input):
        global size, halign, value
        if screen_manager.get_screen('chats').text_input != "":
            now = datetime.datetime.now()
            value = screen_manager.get_screen('chats').text_input.text
            if len(value) < 6:
                size = .22
                halign = "center"
            elif len(value) < 11:
                size = .32
                halign = "center"
            elif len(value) < 16:
                size = .45
                halign = "center"
            elif len(value) < 21:
                size = .58
                halign = "center"
            elif len(value) < 26:
                size = .71
                halign = "center"
            else:
                size = .71
                halign = "left"
            screen_manager.get_screen('chats').chat_list.add_widget(
                Command(text=value, size_hint_x=size, halign=halign))
            Clock.schedule_once(self.response, 3)
            screen_manager.get_screen('chats').text_input.text = ""


    def phq_score(self, checkbox, value, score):
        global phq_total_score
        current_score = score
        if score == 0 and checkbox.state == "down":
            phq_total_score += current_score
        if score == 1 and checkbox.state == "down":
            phq_total_score += current_score
        if score == 2 and checkbox.state == "down":
            phq_total_score += current_score
        if score == 3 and checkbox.state == "down":
            phq_total_score += current_score

    def gad_score(self, checkbox, value, x):
        global gad_total_score
        if x == 0 and checkbox.state == "down":
            gad_total_score += x
        if x == 1 and checkbox.state == "down":
            gad_total_score += x
        if x == 2 and checkbox.state == "down":
            gad_total_score += x
        if x == 3 and checkbox.state == "down":
            gad_total_score += x

    def show_alert_dialog(self):
        phq_score = phq_total_score
        if 1 <= phq_score <= 4:
            test_result = "Tahniah, awak dalam kategori depressi minimum. Saya berharap agar awak dapat terus konsisten dalam menjalani hidup awak! "
        if 5 <= phq_score <= 9:
            test_result = "Anda mengalami depresi di tahap rendah. Saya berharap ia tidak menggangu kehidupan seharian awak. Walaubagaimanapun, saya pasti apa yang awak alami sekarang membuatkan awak rasa tidak selesa. Apa yang boleh okBot syorkan sekarang, bergerak dan ambil masa 10 minit untuk berjalan di luar, rasa angin sepoi di luar."
        if 10 <= phq_score <= 14:
            test_result = "Anda mengalami depresi di tahap serdahana. Saya tahu bukan mudah untuk terima dan alami semua benda ini. Awak rasa sedih, penat tak bermaya, susah nak fokus pada sesuatu benda. Tapi saya nak ingatkan awak satu benda, dengan menggunakan Ok-bot ini ialah salah satu tanda yang awak ingin tolong diri sendiri. Tahniah! Perkara ini bukan mudah tapi awak sudah lakukan satu step kehadapan"
        if 15 <= phq_score <= 19:
            test_result = "Anda mengalami depresi di tahap serdahana teruk. Saya tahu bukan mudah untuk terima dan alami semua benda ini. Awak rasa sedih, penat tak bermaya, susah nak fokus pada sesuatu benda. Tapi saya nak ingatkan awak satu benda, dengan menggunakan Ok-bot ini ialah salah satu tanda yang awak ingin tolong diri sendiri. Tahniah! Perkara ini bukan mudah tapi awak sudah lakukan satu step kehadapan"
        if 20 <= phq_score <= 27:
            test_result = "Anda mengalami depresi di tahap teruk. Saya tahu bukan mudah untuk terima dan alami semua benda ini. Awak rasa sedih, penat tak bermaya, susah nak fokus pada sesuatu benda. Tapi saya nak ingatkan awak satu benda, dengan menggunakan Ok-bot ini ialah salah satu tanda yang awak ingin tolong diri sendiri. Tahniah! Perkara ini bukan mudah tapi awak sudah lakukan satu step kehadapan"
        if not self.dialog:
            self.dialog = MDDialog(
                text="Skor PHQ-9: " + str(phq_score) + "\n" + test_result,
                size_hint=[0.9, None],
                buttons=[
                    MDFlatButton(
                        text="Tutup",
                        on_release=self.close_dialog
                    ),
                ],
            )
        self.dialog.open()

    def show_alert_dialogs(self):
        gad_score = gad_total_score
        if 0 <= gad_score <= 4:
            test_result = "Anda dalam kategori anxiety minimum"
        if 5 <= gad_score <= 9:
            test_result = "Anda mengalami anxiety di tahap rendah"
        if 10 <= gad_score <= 14:
            test_result = "Anda mengalami anxiety di tahap serdahana"
        if 15 <= gad_score <= 21:
            test_result = "Anda mengalami anxiety di tahap teruk"
        if not self.dialogs:
            self.dialogs = MDDialog(
                text="Skor GAD-7: " + str(gad_total_score) + "\n" + test_result,
                size_hint=[0.9, None],
                buttons=[
                    MDFlatButton(
                        text="Tutup",
                        on_release=self.close_dialogs
                    ),
                ],
            )
        self.dialogs.open()

    def close_dialog(self, obj):
        self.dialog.dismiss()

    def close_dialogs(self, obj):
        self.dialogs.dismiss()

    def close_dialog_login(self, obj):
        self.dialog_login.dismiss()

if __name__ == '__main__':
    LabelBase.register(name="Poppins", fn_regular="Fonts/Poppins-Regular.ttf")
    print("okBot is running!")
    print("Type something..")
    OkBot().run()