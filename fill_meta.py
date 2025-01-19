import os
import cv2 as cv
import logging
import numpy as np
import face_recognition
import PySimpleGUI as sg
from typing import Tuple
from db import get_next_id, save_user, save_embedding
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)
FILE_TYPES = [
    ("PNG (*.png)", ["*.png", "*.PNG"]), 
    ("JPG (*.jpg)", ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]), 
    ("All files (*.*)", "*.*")]
ROOT = Path(__file__).resolve().parent
LAYOUT = [
    [
        sg.Text("Имя:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-FIRSTNAME-")
    ],
    [
        sg.Text("Фамилия:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-SECONDNAME-")
    ],
    [
        sg.Text("Статус:", size=(15, 1)),
        sg.Input(default_text="G", size=(30, 1), key="-STANDING-")
    ],
    [
        sg.Text("Специализация:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-MAJOR-")
    ],
    [
        sg.Text("Начало обучения:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-YEAR-")
    ],
    [
        sg.Text("Фотография", size=(15, 1)),
        sg.Input(size=(30, 1), key="-SRC-", disabled=True),
        sg.FileBrowse(file_types=FILE_TYPES, size=(8, 1), button_text="Открыть")
    ],
    [
        sg.Text("Записать в", size=(15, 1)),
        sg.Input(size=(30, 1), key="-DST-", disabled=True, default_text=str(ROOT / "data" / "faces")),
        sg.FolderBrowse(size=(8, 1), button_text="Открыть")
    ],
    [
        sg.Button("Сохранить", size=(12, 1)), sg.Button("Закрыть", size=(12, 1)), sg.Button("Экспортировать в CSV", size=(17, 1)),
    ],
    [
        sg.StatusBar("", size=(0, 1), key='-STATUS-')
    ],
]


def encode(face: np.ndarray) -> np.ndarray:
    # BGR (uses by cv2) в RGB (face_recognition)
    face = cv.cvtColor(face, cv.COLOR_BGR2RGB)

    # кодировка
    encoded = face_recognition.face_encodings(face)[0]
    return encoded


def get_face(src: str, dst: str, sc: int = 2, size: Tuple[int, int] = (216, 216)) -> np.ndarray:
    # открываем изображение с лицами
    img = cv.imread(src)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # создаем HOG дескриптор и загружаем предобученный классификатор для лиц
    hog_face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # детектим лица
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    bboxes = hog_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print("Лиц на изображении обнаружено: %s" % len(bboxes))

    # итерируемся по лицам и находим лицо с максимальной площадью
    margin, max_area = 0, 0
    det_face = None
    im_h, im_w, n_ch = img.shape

    for (x, y, w, h) in bboxes:
        area = w * h
        
        if max_area < area:
            xc, yc = x + w // 2, y + h // 2
            w, h = int(w * sc), int(h * sc)
            x_new, y_new = int(xc - w // 2), int(yc - h // 2)
            margin = max(0, (x_new + w) - im_w, 0 - x_new, (y_new + h) - im_h, 0 - y_new)
            det_face = (x_new, y_new, w, h)
            max_area = area

    if det_face is None:
        print("Лица не найдены.")
        return None

    # вырезаем нужное нам лицо
    ext_img = np.full(shape=(im_h + 2 * margin, im_w + 2 * margin, n_ch), fill_value=255, dtype="uint8")
    ext_img[margin:margin + im_h, margin:margin + im_w, :] = img
    x_new, y_new, w, h = det_face
    x_new += margin
    y_new += margin
    det_face = ext_img[y_new:y_new + h, x_new:x_new + w, :]

    # меняем размер изображения
    det_face = cv.cvtColor(det_face, cv.COLOR_RGB2BGR)
    det_face = cv.resize(det_face, size, interpolation=cv.INTER_LINEAR)

    # сохраняем
    cv.imwrite(dst, det_face)
    return det_face
    
def export_to_csv(data: list, filename: str) -> None:
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def main() -> None:
    window = sg.Window("Форма для заполнения", LAYOUT)
    student_data = []  # Список для хранения данных студентов

    while True:
        event, values = window.read()

        if event == "Закрыть" or event == sg.WIN_CLOSED:
            break

        else:
            # считываем поля
            firstname = values["-FIRSTNAME-"]
            secondname = values["-SECONDNAME-"]
            standing = values["-STANDING-"]
            major = values["-MAJOR-"]
            starting_year = values["-YEAR-"]
            src = values["-SRC-"]
            dst = values["-DST-"]

            if event == "Сохранить":

                # проверяем, что все поля заполнены
                if (isinstance(firstname, str) and len(firstname) > 0) and \
                   (isinstance(secondname, str) and len(secondname) > 0) and \
                   (isinstance(standing, str) and len(standing) == 1) and \
                   (isinstance(major, str) and len(major) > 0) and \
                   (isinstance(starting_year, str) and starting_year.strip().isdecimal()) and \
                   (isinstance(src, str) and os.path.exists(src)):

                    # значения делаем upper case
                    firstname = firstname.strip().upper()
                    secondname = secondname.strip().upper()
                    standing = standing.strip().upper()
                    major = major.strip().upper()
                    starting_year = int(starting_year.strip())

                    # выводим результат
                    LOGGER.debug("Имя: %s" % firstname)
                    LOGGER.debug("Фамилия: %s" % secondname)
                    LOGGER.debug("Статус: %s" % standing)
                    LOGGER.debug("Специализация: %s" % major)
                    LOGGER.debug("Начало обучения: %s" % starting_year)

                    # если нет пути для сохранения, то создаем
                    if not os.path.exists(dst):
                        LOGGER.debug("Создаем папку %s" % dst)
                        os.makedirs(dst)

                    # получаем следующий свободный ID студента
                    LOGGER.debug("Получаем ID студента")
                    id = get_next_id()
                    LOGGER.debug("ID: %s" % id)

                    # собираем метаданные по студенту
                    LOGGER.debug("Собираем метаданные")
                    metadata = dict(
                        id=id,
                        firstname=firstname,
                        secondname=secondname,
                        standing=standing,
                        major=major,
                        starting_year=starting_year,
                    )

                    # получаем эмбеддинги лица
                    LOGGER.debug("Получаем эмбеддинг лица")
                    _, ext = os.path.splitext(src)
                    dst = os.path.join(dst, "%s%s" % (id, ext))
                    face = get_face(src=src, dst=dst)
                    emb = encode(face=face)
                    
                    # записываем в базу
                    save_user(**metadata)
                    save_embedding(id=id, emb=emb)
                    LOGGER.debug("Данные сохранены")

                    state = "Данные сохранены"
                else:
                    state = "Поля не заполнены / запонены не верно"

                window['-STATUS-'].update(state)             
            elif event == "Экспортировать в CSV":
                if student_data:
                    filename = sg.popup_get_file('Сохранить как', save_as=True, no_window=True, file_types=(('CSV Files', '*.csv'),))
                    if filename:
                        export_to_csv(student_data, filename)
                        sg.popup("Данные успешно экспортированы в CSV.")
                else:
                    sg.popup("Нет данных для экспорта.")
    window.close()
    return


if __name__ == "__main__":
    main()
