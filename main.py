import os
import re
import time
import cvzone
import logging
import cv2 as cv
import numpy as np
import face_recognition
from pathlib import Path
from collections import deque
from db import read_embeddings, mark_users


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = str(ROOT / "data" / "background-1.png")
PHOTO_PATH = str(ROOT / "data" / "faces")
OUTPUT = str(ROOT / "output.mp4")
SCALE = 0.25
TOL = 0.55
MAX_Q = 5


def make_card(metadata: dict, photo_loc: str, width: int, height: int, n_channels: int = 3) -> np.ndarray:
    card = np.full(shape=(height, width, n_channels), fill_value=255, dtype=np.uint8)

    # read photo
    if isinstance(photo_loc, str) and os.path.exists(photo_loc):
        photo = cv.imread(photo_loc)
        photo = cv.resize(src=photo, dsize=(height, height), interpolation=cv.INTER_CUBIC)
        card[:, :height] = photo

    # adding text
    # name
    firstname, secondname = metadata.get("firstname"), metadata.get("secondname")
    firstname =  firstname.strip().capitalize() if isinstance(firstname, str) else ""
    secondname = secondname.strip().capitalize() if isinstance(secondname, str) else ""
    name = "%s %s" % (firstname, secondname)
    name = name.strip()
    cv.putText(img=card, text=name, org=(height + 15, 35), fontFace=cv.FONT_HERSHEY_COMPLEX, 
               fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    
    anchor = 65
    space = 20
    # major
    major = metadata.get("major")
    major = major.strip().capitalize() if isinstance(major, str) else "-"
    cv.putText(img=card, text="Специальность:", org=(height + 15, anchor), fontFace=cv.FONT_HERSHEY_COMPLEX, 
               fontScale=0.5, color=(0, 0, 0), thickness=1)
    cv.putText(img=card, text=major, org=(280, anchor), 
               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    anchor += space

    # attendances
    n_att = metadata.get("n_att")
    n_att = n_att if isinstance(n_att, int) else 0
    cv.putText(img=card, text="Посещаемость:", org=(height + 15, anchor), fontFace=cv.FONT_HERSHEY_COMPLEX, 
               fontScale=0.5, color=(0, 0, 0), thickness=1)
    cv.putText(img=card, text="%s" % n_att, org=(280, anchor), 
               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    anchor += space

    # year
    year = metadata.get("starting_year")
    year = year if isinstance(year, int) else "-"
    cv.putText(img=card, text="Год:", org=(height + 15, anchor), fontFace=cv.FONT_HERSHEY_COMPLEX, 
               fontScale=0.5, color=(0, 0, 0), thickness=1)
    cv.putText(img=card, text="%s" % year, org=(280, anchor), 
               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    anchor += space

    # status
    status = metadata.get("status")
    status = status if isinstance(status, str) else "-"
    cv.putText(img=card, text="Статус:", org=(height + 15, anchor), fontFace=cv.FONT_HERSHEY_COMPLEX, 
               fontScale=0.5, color=(0, 0, 0), thickness=1)
    cv.putText(img=card, text="%s" % status, org=(280, anchor), 
               fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    return card


def add_cards(metadata: dict, background: np.ndarray, photo_loc: str = "./data/faces/", pos=[(752, -1), (-1, -1)], n_cards: int = 5, margin: int = 5) -> np.ndarray:
    bg_height, bg_width, n_channels = background.shape
    left, bottom = pos[0]
    right, top   = pos[1]
    if left   == -1: left   = 0
    if bottom == -1: bottom = 0
    if right  == -1: right  = bg_width
    if top    == -1: top    = bg_height

    card_width, card_height = (right - left) - 2*margin, ((top - bottom) - (n_cards + 1)*margin)//n_cards

    for id, meta in metadata.items():
        if bottom + 2*margin + card_height > top:
            break

        for f in os.listdir(photo_loc):
            if re.fullmatch("%s\.(?:jpg)|(?:png)|(?:jpeg)" % id, f, re.IGNORECASE):
                f = os.path.join(photo_loc, f)
                break
        else:
            f = None
        card = make_card(meta, f, card_width, card_height, n_channels=n_channels)
        background[bottom + margin: bottom + margin + card_height, left + margin:left + margin + card_width] = card
        bottom += margin + card_height

    return background


def compare_embs(face_encodings_to_check: list, hidden_size: int = 128) -> int:
    # load all embeddings
    LOGGER.info("Loading embeddings")
    known_faces          = read_embeddings(hidden_size=hidden_size)
    known_face_encodings = []
    metadata             = []
    for data in known_faces:
        encoding = data[:hidden_size]
        meta     = data[hidden_size:]
        known_face_encodings.append(encoding)
        metadata.append(meta)
    
    LOGGER.info("Loaded %s embeddings" % len(known_faces))

    for face_encoding_to_check in face_encodings_to_check:
        matches = face_recognition.compare_faces(
            known_face_encodings=known_face_encodings, face_encoding_to_check=face_encoding_to_check, tolerance=TOL)
        dist = face_recognition.face_distance(face_encodings=known_face_encodings, face_to_compare=face_encoding_to_check)

        # ищем ближайшее лицо
        clst_idx = np.argmin(dist)
        if matches[clst_idx]:
            face_meta = metadata[clst_idx]
        else:
            face_meta = -1
        yield face_meta


def main() -> None:
    cap = cv.VideoCapture(0)  # create webcam object
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set hight

    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    out = cv.VideoWriter(OUTPUT, fourcc, 30.0, (1280, 720))

    q = deque()
    while (cap.isOpened()):
        # read frame from webcap object
        success, frame = cap.read()

        if not success:
            continue

        # loading images (background and for modes)
        background = cv.imread(BACKGROUND_PATH)

        # resize image from webcam to save computation resources
        # and convert to RGB from BGR
        frame_s = cv.resize(frame, (0, 0), None, SCALE, SCALE)
        frame_s = cv.cvtColor(frame_s, cv.COLOR_BGR2RGB)

        # находим все лица на изображении и получаем для них encodings
        face_curr_locations     = face_recognition.face_locations(frame_s)
        face_encodings_to_check = face_recognition.face_encodings(frame_s, known_face_locations=face_curr_locations)
        LOGGER.info("Обнаружено лиц: %s" % len(face_encodings_to_check))

        # удаляем старые
        for _, _, t in list(q):
            t_delta = (time.time() - t) > 10
            if t_delta:
                q.popleft()

        bboxes_to_plot = zip(compare_embs(face_encodings_to_check=face_encodings_to_check), face_curr_locations)
        for face_meta, bbox in bboxes_to_plot:
            bottom, right, top, left = bbox
            x1, y1 = round(left / SCALE), round(bottom / SCALE)
            w, h   = round((right - left) / SCALE), round((top - bottom) / SCALE)
            bbox   = (x1, y1, w, h)

            if face_meta == -1:
                text  = "Unknown"
                color = (0, 0, 255)

            else:
                id, firstname, secondname, major, starting_year, n_att = face_meta
                face_meta = dict(firstname=firstname, secondname=secondname, major=major, 
                                 starting_year=starting_year, n_att=n_att)
                firstname  = firstname.strip().capitalize()  if isinstance(firstname, str) else ""
                secondname = secondname.strip().capitalize() if isinstance(firstname, str) else ""
                text = "%s %s" % (firstname, secondname)
                text = text.strip()
                color = (0, 255, 0)

                # отметим
                marked = mark_users(ids=id)
                if marked: face_meta["status"] = "Отмечен"
                else:      face_meta["status"] = "Уже отмечен"

                # добавляем в очередь, если нет
                for i in range(len(q)):
                    if q[i][0] == id:
                        q[i][2] = time.time()
                        break
                else:
                    q.append([id, face_meta, time.time()])
                    if len(q) > MAX_Q:
                        q.popleft()

            cvzone.cornerRect(img=frame, bbox=bbox, colorC=color)
            cvzone.putTextRect(img=frame, text=text, pos=(max(0, x1), max(35, y1 - 10)), 
                               scale=1, thickness=1, offset=3, colorR=color, colorT=(0, 0, 0))
        cards = {id: meta for id, meta, _ in q}    
        background = add_cards(cards, background, n_cards=MAX_Q, photo_loc=PHOTO_PATH)

        # overlay background with webcam 
        background[162:162 + 480, 55:55 + 640] = frame

        out.write(background)
        cv.imshow("Face Attendance", background)

        # exit due to ESC key
        if cv.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
    