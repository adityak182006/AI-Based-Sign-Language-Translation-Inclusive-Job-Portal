# =========================================================
# ===================== BASIC SETUP =======================
# =========================================================

import os
import cv2
import time
import pickle
import numpy as np
import warnings

from flask import (
    Flask, render_template,
    Response, jsonify, request
)

import mediapipe as mp
import enchant

warnings.filterwarnings("ignore")

print("✅ Starting backend initialization...")

# =========================================================
# ========================= CONFIG ========================
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIGN_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMG_SIZE = (300, 300)

CHAR_DELAY = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# ====================== LOAD SIGNS =======================
# =========================================================

def load_signs():
    signs = {}
    if not os.path.exists(SIGN_DIR):
        raise RuntimeError("❌ images folder not found")

    for f in os.listdir(SIGN_DIR):
        if f.lower().endswith(".png"):
            signs[f.split(".")[0].upper()] = os.path.join(SIGN_DIR, f)

    if not signs:
        raise RuntimeError("❌ No sign images found in images/")

    return signs

SIGNS = load_signs()
print(f"✅ Loaded {len(SIGNS)} sign images")

# =========================================================
# ========================= FLASK =========================
# =========================================================

app = Flask(__name__)

# ---------------- TEST ROUTE ----------------

@app.route("/ping")
def ping():
    return "Flask is running"

# =========================================================
# ====================== LOAD MODEL =======================
# =========================================================

with open("model.p", "rb") as f:
    model = pickle.load(f)["model"]

print("✅ ML model loaded")

dictionary = enchant.Dict("en_US")

# =========================================================
# ======================= MEDIAPIPE =======================
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("✅ MediaPipe initialized")

# =========================================================
# ========================= LABELS ========================
# =========================================================

labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',
    8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',
    15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',
    22:'W',23:'X',24:'Y',25:'Z',
    26:'Hello',27:'Done',28:'Thank You',
    29:'I Love You',30:'Sorry',31:'Please',32:'You are welcome'
}

# =========================================================
# =========================== STATE =======================
# =========================================================

current_word = ""
paragraph = ""
word_stack = []
suggestions = []
latest_payload = {}

paused = False
stable_char = None
stable_since = 0
last_accepted_char = None

# =========================================================
# ========================== HELPERS ======================
# =========================================================

def draw_subtitle(frame, text):
    if not text:
        return frame

    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    size, _ = cv2.getTextSize(text, font, 0.9, 2)
    x = (w - size[0]) // 2
    y = h - 20

    cv2.putText(frame, text, (x+2, y+2), font, 0.9, (0,0,0), 4)
    cv2.putText(frame, text, (x, y), font, 0.9, (255,255,255), 2)

    return frame

def get_next_untitled_video():
    i = 1
    while True:
        path = os.path.join(OUTPUT_DIR, f"untitled_{i}.mp4")
        if not os.path.exists(path):
            return path
        i += 1

# =========================================================
# ========================== ROUTES =======================
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    return jsonify(latest_payload)

@app.route("/pause", methods=["POST"])
def pause():
    global paused
    paused = request.json.get("paused", False)
    return ("", 204)

@app.route("/clear", methods=["POST"])
def clear():
    global current_word, paragraph, word_stack
    global stable_char, stable_since, last_accepted_char

    current_word = ""
    paragraph = ""
    word_stack = []
    stable_char = None
    stable_since = 0
    last_accepted_char = None

    return ("", 204)

@app.route("/undo", methods=["POST"])
def undo():
    global paragraph, word_stack
    if word_stack:
        word_stack.pop()
        paragraph = " ".join(word_stack) + " "
    return ("", 204)

@app.route("/select_word", methods=["POST"])
def select_word():
    global paragraph, current_word, word_stack
    word = request.json.get("word")
    if word:
        word_stack.append(word)
        paragraph = " ".join(word_stack) + " "
        current_word = ""
    return ("", 204)

# =========================================================
# ====================== SIGN → TEXT ======================
# =========================================================

def generate_camera_frames():
    global current_word, paragraph, suggestions, latest_payload
    global paused, stable_char, stable_since, last_accepted_char

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if not paused:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected = None

            if result.multi_hand_landmarks:
                for hnd in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hnd, mp_hands.HAND_CONNECTIONS)

                    xs, ys, feats = [], [], []
                    for lm in hnd.landmark:
                        xs.append(lm.x)
                        ys.append(lm.y)
                    for lm in hnd.landmark:
                        feats.extend([lm.x - min(xs), lm.y - min(ys)])

                    if len(feats) == model.n_features_in_:
                        pred = model.predict([np.array(feats)])
                        detected = labels_dict[int(pred[0])]

            now = time.time()

            if detected is None:
                stable_char = None
                stable_since = 0
                last_accepted_char = None
            elif detected == last_accepted_char:
                pass
            else:
                if detected != stable_char:
                    stable_char = detected
                    stable_since = now
                elif now - stable_since > CHAR_DELAY:
                    last_accepted_char = detected
                    stable_char = None
                    stable_since = 0

                    if detected == "Done":
                        if current_word:
                            word_stack.append(current_word)
                        paragraph = " ".join(word_stack) + ". "
                        current_word = ""
                    elif len(detected) == 1:
                        current_word += detected
                    else:
                        word_stack.append(detected)
                        paragraph = " ".join(word_stack) + " "

            suggestions = dictionary.suggest(current_word)[:5] if len(current_word) >= 2 else []

            latest_payload = {
                "char": detected or "",
                "word": current_word,
                "paragraph": paragraph.strip(),
                "suggestions": suggestions
            }

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# =========================================================
# =================== TEXT → SIGN STREAM ==================
# =========================================================

@app.route("/text_to_sign_stream")
def text_to_sign_stream():
    text = request.args.get("text", "").upper()
    fps = int(request.args.get("fps", 2))
    delay = 1.0 / max(fps, 1)

    def generate():
        for word in text.split():
            for ch in word:
                if ch in SIGNS:
                    img = cv2.imread(SIGNS[ch])
                    img = cv2.resize(img, IMG_SIZE)
                    img = draw_subtitle(img, word)
                    _, buffer = cv2.imencode(".jpg", img)
                    yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" +
                           buffer.tobytes() + b"\r\n")
                    time.sleep(delay)

            pause = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
            pause = draw_subtitle(pause, word)
            _, buffer = cv2.imencode(".jpg", pause)
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
            time.sleep(delay)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# =========================================================
# ================= SAVE TEXT → SIGN VIDEO (MP4) ==========
# =========================================================

@app.route("/save_text_to_sign", methods=["POST"])
def save_text_to_sign():
    data = request.json
    text = data.get("text", "").upper()
    fps = int(data.get("fps", 2))

    path = get_next_untitled_video()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, IMG_SIZE)

    for word in text.split():
        for ch in word:
            if ch in SIGNS:
                img = cv2.imread(SIGNS[ch])
                img = cv2.resize(img, IMG_SIZE)
                img = draw_subtitle(img, word)
                writer.write(img)

        pause = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        pause = draw_subtitle(pause, word)
        writer.write(pause)

    writer.release()

    return jsonify({
        "saved": True,
        "file": os.path.basename(path)
    })

# =========================================================
# =========================== START =======================
# =========================================================

if __name__ == "__main__":
    print("✅ Flask starting on http://127.0.0.1:8000")
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
        use_reloader=False
    )
