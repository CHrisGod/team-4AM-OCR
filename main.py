class MyImage:
    def _init_(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def _str_(self):
        return self.__name
import cv2
import pytesseract
import time
from PIL import Image
import numpy as np
import easygui
import tkinter
window_main = tkinter.Tk(className='SPECATHON T-11 PROJECT', )
window_main.geometry("400x200")
def close_window ():
    window_main.destroy()
def program ():
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    path = easygui.fileopenbox()
    imageName = path
    img = cv2.imread(imageName)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(pytesseract.image_to_string(img))
    s = MyImage(imageName)
    str(s)
    moment = time.strftime("-%Y-%b-%d_%H%M_%S", time.localtime())
    file = open(str(s) + moment + '.txt', "w+")
    file.close()
    file = open(str(s) + moment + '.txt', "a")
    text = pytesseract.image_to_string(img)
    file.write(text)
    file.close
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hImg, wImg, _ = img.shape
    boxes = pytesseract.image_to_data(img)
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            if len(b) == 12:
                       x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                       cv2.rectangle(img, (x, y), (w + x, h + y), (1, 200, 255), 3)
                       cv2.putText(img, b[11], (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (25, 25, 255), 2)
    print("TXT file is created")
    img = cv2.resize(img, (1080, 650))
    cv2.imshow('Result', img)

button_submit = tkinter.Button(window_main, text ="Upload Image File", command=program)
button_submit.config(width = 20, height=2)
button_submit.pack()
button_exit = tkinter.Button(window_main, text ="Exit", command=close_window)
button_exit.config(width=20, height=2)
button_exit.pack()
window_main.mainloop()