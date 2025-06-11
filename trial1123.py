import cv2
import numpy as np
import os
from datetime import datetime
import time
import requests
from collections import deque
from scipy.spatial import distance as dist
import signal
import sys
from flask import Flask, render_template_string, Response, send_from_directory, request, redirect, url_for, session, flash
import threading
import json
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

# ==== Signal Handler for Ctrl+C ====
def signal_handler(sig, frame):
    print("Received Ctrl+C, initiating shutdown...")
    raise SystemExit

signal.signal(signal.SIGINT, signal_handler)

# ==== Telegram Setup ====
TELEGRAM_BOT_TOKEN = "7959551214:AAF4pUbQuItuttpMekUTxdD3EaivE41SDMI"
TELEGRAM_CHAT_ID = "6602085152"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Telegram message sent successfully")
        else:
            print(f"Telegram message failed: {response.text}")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                print("Telegram photo sent successfully")
            else:
                print(f"Telegram photo failed: {response.text}")
    except Exception as e:
        print(f"Failed to send Telegram photo: {e}")

# ==== YOLO Setup ====
try:
    yolo = cv2.dnn.readNet("yolov2-tiny.weights", "yolov2-tiny.cfg")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

try:
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()
except Exception as e:
    print(f"Error loading COCO names: {e}")
    sys.exit(1)

SAVE_DIR = "detections"
NOTIFICATION_LOG = "notifications.txt"
NOTIFIED_PERSONS_FILE = "notified_persons.json"
USERS_FILE = "users.json"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== User Management ====
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading users: {e}")
        return {}

def save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        print(f"Error saving users: {e}")

# ==== Write Notification to Text File ====
def log_notification(message, image_filename):
    try:
        with open(NOTIFICATION_LOG, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message} | {image_filename}\n")
    except Exception as e:
        print(f"Failed to log notification: {e}")

# ==== Load/Save Notified Persons ====
def load_notified_persons():
    try:
        if os.path.exists(NOTIFIED_PERSONS_FILE):
            with open(NOTIFIED_PERSONS_FILE, "r") as f:
                data = json.load(f)
                return [set(data.get(f"cam_{i+1}", [])) for i in range(len(CAMERA_LIST))]
        return [set() for _ in CAMERA_LIST]
    except Exception as e:
        print(f"Error loading notified persons: {e}")
        return [set() for _ in CAMERA_LIST]

def save_notified_persons(notified_persons):
    try:
        data = {f"cam_{i+1}": list(notified_persons[i]) for i in range(len(CAMERA_LIST))}
        with open(NOTIFIED_PERSONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving notified persons: {e}")

# ==== Smart One-Person-One-ID Tracker ====
class SmartPersonTracker:
    def __init__(self, max_disappeared=300, max_history=40, match_radius=150):
        self.person_id = 1
        self.persons = []
        self.max_disappeared = max_disappeared
        self.max_history = max_history
        self.match_radius = match_radius
        self.load_last_person_id()

    def load_last_person_id(self):
        try:
            if os.path.exists(NOTIFIED_PERSONS_FILE):
                with open(NOTIFIED_PERSONS_FILE, "r") as f:
                    data = json.load(f)
                    all_ids = []
                    for cam_key in data:
                        all_ids.extend(int(pid) for pid in data[cam_key] if pid.isdigit())
                    if all_ids:
                        self.person_id = max(all_ids) + 1
        except Exception as e:
            print(f"Error loading last person ID: {e}")

    def update(self, rects):
        objects_out = {}
        current_centroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            current_centroids.append((cX, cY, x, y, w, h))
        matched = []
        for cX, cY, x, y, w, h in current_centroids:
            found = False
            for person in self.persons:
                if (datetime.now() - person['last_seen']).total_seconds() > self.max_disappeared:
                    continue
                prev_cX, prev_cY = person['trace'][-1] if person['trace'] else person['centroid']
                if dist.euclidean([cX, cY], [prev_cX, prev_cY]) < self.match_radius:
                    person['centroid'] = (cX, cY)
                    person['trace'].append((cX, cY))
                    if len(person['trace']) > self.max_history:
                        person['trace'].popleft()
                    person['last_seen'] = datetime.now()
                    found = True
                    matched.append(person['id'])
                    objects_out[person['id']] = (cX, cY, x, y, w, h)
                    break
            if not found:
                new_person = {
                    'id': self.person_id,
                    'centroid': (cX, cY),
                    'trace': deque([(cX, cY)], maxlen=self.max_history),
                    'last_seen': datetime.now()
                }
                self.persons.append(new_person)
                objects_out[self.person_id] = (cX, cY, x, y, w, h)
                matched.append(self.person_id)
                self.person_id += 1
        self.persons = [p for p in self.persons if (datetime.now() - p['last_seen']).total_seconds() < self.max_disappeared]
        return objects_out

# ===================== CAMERA CONFIG ========================
CAMERA_LIST = [
    {
        "name": "Camera - 1 live  ",
        "footer": "PRIMINE SOFTWARE PVT.LTD",
        "color": "#f7cac9",
        "rtsp_url": "rtsp://admin:ADMIN%40123@192.168.29.163:554/cam/realmonitor?channel=2&subtype=0"
    },
    {
        "name": "Camera - 2 live ",
        "footer": "PRIMINE SOFTWARE PVT.LTD",
        "color": "#dbe6e4",
        "rtsp_url": "rtsp://admin:ADMIN%40123@192.168.29.163:554/cam/realmonitor?channel=1&subtype=0"
    }
]
LIVE_FRAMES = [None for _ in CAMERA_LIST]

def update_live_frames(idx, frame):
    LIVE_FRAMES[idx] = frame

def gen_camera(idx):
    while True:
        frame = LIVE_FRAMES[idx]
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)

# ===================== FLASK SETUP WITH AUTHENTICATION ========================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a secure key in production

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash("Please log in to access the dashboard.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for('register'))

        users = load_users()
        if username in users:
            flash("Username already exists. Please choose another.", "error")
            return redirect(url_for('register'))

        users[username] = generate_password_hash(password)
        save_users(users)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Register - Smart Surveillance Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Poppins', Arial, sans-serif;
      background: linear-gradient(120deg, #141e30 0%, #243b55 100%);
      color: #fff;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 1rem;
    }
    .auth-container {
      background: rgba(0, 0, 0, 0.7);
      padding: 2.5rem 2rem 2rem 2rem;
      border-radius: 18px;
      width: 100%;
      max-width: 420px;
      box-shadow: 0 10px 28px 0 rgba(0,0,0,0.28), 0 2px 4px 0 rgba(0,0,0,0.18);
      position: relative;
      overflow: hidden;
    }
    .auth-container::before {
      content: "";
      position: absolute;
      left: -60px;
      top: -60px;
      width: 140px;
      height: 140px;
      background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
      opacity: 0.15;
      border-radius: 50%;
      z-index: 0;
    }
    .auth-container::after {
      content: "";
      position: absolute;
      right: -60px;
      bottom: -60px;
      width: 130px;
      height: 130px;
      background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
      opacity: 0.10;
      border-radius: 50%;
      z-index: 0;
    }
    .auth-content {
      position: relative;
      z-index: 1;
    }
    h2 {
      text-align: center;
      margin-bottom: 1.6rem;
      font-size: 2rem;
      font-weight: 600;
      letter-spacing: 1px;
      color: #00c6ff;
      text-shadow: 0 2px 16px #0072ff55;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }
    label {
      font-weight: 500;
      margin-bottom: 0.35rem;
      color: #c0e7fa;
      letter-spacing: 0.2px;
      font-size: 1rem;
    }
    input {
      padding: 0.7rem 1rem;
      margin-bottom: 0.4rem;
      border: none;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.12);
      color: #fff;
      font-size: 1rem;
      font-family: inherit;
      box-shadow: 0 2px 8px #0072ff22;
      transition: background 0.2s, box-shadow 0.2s;
    }
    input:focus {
      outline: none;
      background: rgba(255, 255, 255, 0.20);
      box-shadow: 0 4px 20px #00c6ff33;
    }
    button {
      margin-top: 0.7rem;
      background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
      color: white;
      border: none;
      padding: 0.9rem 0;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 1px;
      box-shadow: 0 4px 16px #00c6ff44;
      transition: background 0.3s, transform 0.2s;
    }
    button:hover {
      background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
      transform: scale(1.03);
      box-shadow: 0 8px 32px #00c6ff66;
    }
    .message {
      text-align: center;
      margin-bottom: 1rem;
      font-size: 1rem;
      font-weight: 500;
    }
    .message.success {
      color: #00ff9f;
    }
    .message.error {
      color: #ff4444;
    }
    .link {
      text-align: center;
      margin-top: 1.2rem;
    }
    .link a {
      color: #00c6ff;
      text-decoration: none;
      transition: color 0.3s;
      font-weight: 500;
    }
    .link a:hover {
      color: #0072ff;
      text-decoration: underline;
    }
    @media (max-width: 500px) {
      .auth-container {
        padding: 1.2rem 0.7rem 1.2rem 0.7rem;
        border-radius: 12px;
      }
      h2 {
        font-size: 1.4rem;
      }
    }
  </style>
</head>
<body>
  <div class="auth-container">
    <div class="auth-content">
      <h2>Register</h2>
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="message {{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
      <form method="POST">
        <label for="fullname">Full Name:</label>
        <input type="text" id="fullname" name="fullname" placeholder="Enter your full name" required>
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" placeholder="Choose a username" required>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" placeholder="Create a password" required>
        <button type="submit">Register</button>
      </form>
      <div class="link">
        <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
      </div>
    </div>
  </div>
</body>
</html>
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        users = load_users()
        if username in users and check_password_hash(users[username], password):
            session['logged_in'] = True
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "error")

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login - Smart Surveillance Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Poppins', Arial, sans-serif;
      background: linear-gradient(120deg, #141e30 0%, #243b55 100%);
      color: #fff;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 1rem;
      overflow-x: hidden;
    }
    .auth-container {
      background: rgba(0, 0, 0, 0.7);
      padding: 2.5rem 2rem 2rem 2rem;
      border-radius: 18px;
      width: 100%;
      max-width: 420px;
      box-shadow: 0 10px 28px 0 rgba(0,0,0,0.28), 0 2px 4px 0 rgba(0,0,0,0.18);
      position: relative;
      overflow: hidden;
      z-index: 2;
    }
    .auth-container::before {
      content: "";
      position: absolute;
      left: -60px;
      top: -60px;
      width: 140px;
      height: 140px;
      background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
      opacity: 0.15;
      border-radius: 50%;
      z-index: 0;
    }
    .auth-container::after {
      content: "";
      position: absolute;
      right: -60px;
      bottom: -60px;
      width: 130px;
      height: 130px;
      background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
      opacity: 0.10;
      border-radius: 50%;
      z-index: 0;
    }
    .auth-content {
      position: relative;
      z-index: 1;
    }
    h2 {
      text-align: center;
      margin-bottom: 1.6rem;
      font-size: 2rem;
      font-weight: 600;
      letter-spacing: 1px;
      color: #00c6ff;
      text-shadow: 0 2px 16px #0072ff55;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }
    label {
      font-weight: 500;
      margin-bottom: 0.35rem;
      color: #c0e7fa;
      letter-spacing: 0.2px;
      font-size: 1rem;
    }
    input {
      padding: 0.7rem 1rem;
      margin-bottom: 0.4rem;
      border: none;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.12);
      color: #fff;
      font-size: 1rem;
      font-family: inherit;
      box-shadow: 0 2px 8px #0072ff22;
      transition: background 0.2s, box-shadow 0.2s;
    }
    input:focus {
      outline: none;
      background: rgba(255, 255, 255, 0.20);
      box-shadow: 0 4px 20px #00c6ff33;
    }
    button {
      margin-top: 0.7rem;
      background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
      color: white;
      border: none;
      padding: 0.9rem 0;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 1px;
      box-shadow: 0 4px 16px #00c6ff44;
      transition: background 0.3s, transform 0.2s;
    }
    button:hover {
      background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
      transform: scale(1.03);
      box-shadow: 0 8px 32px #00c6ff66;
    }
    .message {
      text-align: center;
      margin-bottom: 1rem;
      font-size: 1rem;
      font-weight: 500;
    }
    .message.success {
      color: #00ff9f;
    }
    .message.error {
      color: #ff4444;
    }
    .link {
      text-align: center;
      margin-top: 1.2rem;
    }
    .link a {
      color: #00c6ff;
      text-decoration: none;
      transition: color 0.3s;
      font-weight: 500;
    }
    .link a:hover {
      color: #0072ff;
      text-decoration: underline;
    }
    @media (max-width: 500px) {
      .auth-container {
        padding: 1.2rem 0.7rem 1.2rem 0.7rem;
        border-radius: 12px;
      }
      h2 {
        font-size: 1.4rem;
      }
    }
  </style>
</head>
<body>
  <div class="auth-container">
    <div class="auth-content">
      <h2>Login</h2>
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="message {{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
      <form method="POST">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" placeholder="Enter your username" required>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" placeholder="Enter your password" required>
        <button type="submit">Login</button>
      </form>
      <div class="link">
        <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
      </div>
    </div>
  </div>
</body>
</html>
    """)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route("/")
@login_required
def dashboard():
    detections = []
    for fname in sorted(os.listdir(SAVE_DIR), reverse=True):
        if not fname.endswith(".jpg"):
            continue
        try:
            parts = fname.replace(".jpg", "").split("_")
            timestamp = parts[1]
            person_id = parts[3].replace("ID", "")
            camera_id = parts[4].replace("cam", "")
            dt_fmt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            dt_str = dt_fmt.strftime("%d %b %Y, %H:%M:%S")
            detections.append({
                "img": fname,
                "timestamp": dt_str,
                "person_id": person_id,
                "camera": f"Camera {camera_id}"
            })
        except Exception as e:
            print(f"Error parsing detection file {fname}: {e}")
            continue

    notifications = []
    try:
        if os.path.exists(NOTIFICATION_LOG):
            with open(NOTIFICATION_LOG, "r") as f:
                lines = f.read().splitlines()
                for line in lines[-10:]:
                    try:
                        timestamp, rest = line.split("] ", 1)
                        timestamp = timestamp[1:]
                        if " | " in rest:
                            message, image_filename = rest.split(" | ")
                            notifications.append({
                                "timestamp": timestamp,
                                "message": message,
                                "image": image_filename
                            })
                        else:
                            notifications.append({
                                "timestamp": timestamp,
                                "message": rest,
                                "image": None
                            })
                    except Exception as e:
                        print(f"Error parsing notification: {line}, {e}")
                        continue
    except Exception as e:
        print(f"Error reading notifications: {e}")

    return render_template_string("""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Surveillance Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --background-dark: linear-gradient(to right, #141e30, #243b55);
      --background-light: linear-gradient(to right, #e9ebf0, #dbe6e4);
      --header-dark: rgba(0, 0, 0, 0.6);
      --header-light: #edf2f7;
      --text-dark: #fff;
      --text-light: #222;
      --card-bg-dark: rgba(255,255,255,0.08);
      --card-bg-light: #f5f7fa;
      --footer-dark: #111;
      --footer-light: #111;
      --primary-dark: #00c6ff;
      --primary-light: #0052cc;
      --table-bg-dark: rgba(255, 255, 255, 0.07);
      --table-bg-light: #f1f4fa;
      --table-header-light: #dbe6e4;
      --error-bg: #fcf6f6;
      --error-border: #ff4444;
      --offline-bg: #fff0f0;
      --offline-border: #ff4444;
    }
    html[data-theme="dark"] body {
      background: var(--background-dark);
      color: var(--text-dark);
    }
    html[data-theme="light"] body {
      background: var(--background-light);
      color: var(--text-light);
    }
    body {
      font-family: 'Poppins', Arial, sans-serif;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      padding: 1.5rem;
      transition: background 0.4s, color 0.4s;
    }
    header {
      background-color: var(--header-dark);
      border-radius: 12px;
      margin-bottom: 1.5rem;
      transition: background 0.4s, color 0.4s;
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 64px;
      padding: 1rem 2rem;
      position: relative;
      flex-wrap: wrap;
      gap: 1rem;
    }
    html[data-theme="light"] header {
      background: var(--header-light);
      color: var(--text-light);
      border-bottom: 2px solid #c9d6e3;
    }
    .header-section {
      display: flex;
      align-items: center;
      gap: 1rem;
      flex: 1;
      min-width: 0;
    }
    .header-center {
      flex: 2;
      min-width: 0;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .dashboard-title {
      font-size: 1.5rem;
      font-weight: 600;
      color: inherit;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      text-align: center;
      width: 100%;
    }
    .user-info {
      font-size: 1rem;
      white-space: nowrap;
      color: inherit;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.8rem;
      min-width: 0;
    }
    .toggle-theme-container {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .theme-switch {
      display: inline-flex;
      align-items: center;
      cursor: pointer;
      user-select: none;
    }
    .theme-switch input {
      display: none;
    }
    .slider {
      width: 42px;
      height: 22px;
      background: #666;
      border-radius: 16px;
      position: relative;
      margin-left: 8px;
      transition: background 0.3s;
      display: inline-block;
      flex-shrink: 0;
    }
    .slider:before {
      content: "";
      position: absolute;
      width: 18px;
      height: 18px;
      background: #fff;
      border-radius: 50%;
      top: 2px;
      left: 2px;
      transition: transform 0.3s, background 0.3s;
      box-shadow: 0 2px 6px #0002;
    }
    .theme-switch input:checked + .slider {
      background: #00c6ff;
    }
    .theme-switch input:checked + .slider:before {
      transform: translateX(20px);
      background: #222;
    }
    .toggle-label {
      font-size: 0.95rem;
      color: #aaa;
      user-select: none;
      margin-right: 0.2rem;
    }
    html[data-theme="light"] .toggle-label { color: #222; }
    .logout-link {
      font-size: 1rem;
      color: #00c6ff;
      text-decoration: none;
      transition: color 0.3s;
      margin-left: 0.5rem;
      margin-right: 0.5rem;
      white-space: nowrap;
    }
    .logout-link:hover {
 PRODUCTION
      color: #0072ff;
      text-shadow: 0 0 6px #00c6ff88;
    }
    .bell-icon {
      cursor: pointer;
      font-size: 1.5rem;
      background: #00c6ff;
      border-radius: 50%;
      padding: 0.5rem;
      transition: background 0.3s, box-shadow 0.3s;
      box-shadow: 0 0 0 0 #00c6ff;
      animation: pulse 2s infinite;
      margin-right: 0.2rem;
      margin-left: 0.2rem;
    }
    .bell-icon:hover {
      background: #0072ff;
      box-shadow: 0 0 8px 3px #00c6ff88;
      animation: none;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 #00c6ff55; }
      70% { box-shadow: 0 0 0 8px transparent; }
      100% { box-shadow: 0 0 0 0 transparent; }
    }
    .notification-modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.8);
      z-index: 1000;
      overflow-y: auto;
      padding: 1rem;
    }
    .notification-content {
      background: rgba(0, 0, 0, 0.7);
      border-radius: 15px;
      padding: 1.5rem;
      max-width: 95vw;
      width: 100%;
      max-height: 80vh;
      overflow-y: auto;
      margin: 5vh auto;
      position: relative;
      transition: background 0.3s;
    }
    html[data-theme="light"] .notification-content {
      background: #fff;
      color: #222;
    }
    .close-button {
      position: sticky;
      position: -webkit-sticky;
      top: 0.6rem;
      float: right;
      z-index: 10;
      right: 0.5rem;
      font-size: 1.5rem;
      cursor: pointer;
      color: #fff;
      background: #ff4444;
      border-radius: 50%;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.3s;
      box-shadow: 0 2px 8px #1116;
    }
    .close-button:hover {
      background: #cc0000;
    }
    .controls {
      text-align: center;
      padding: 1rem;
      background: rgba(0, 0, 0, 0.5);
      border-radius: 12px;
      margin-bottom: 1.5rem;
      transition: background 0.4s;
    }
    .controls button {
      background: #00c6ff;
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      margin: 0.5rem;
      font-size: 1rem;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s, box-shadow 0.3s, transform 0.2s;
      box-shadow: 0 2px 8px #00c6ff44;
    }
    .controls button:hover {
      background: #0072ff;
      color: #fff;
      box-shadow: 0 4px 16px #00c6ff88;
      transform: translateY(-2px) scale(1.06);
    }
    .camera-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      padding: 1.5rem;
      justify-content: center;
      align-items: flex-start;
      width: 100%;
      max-width: 100vw;
      box-sizing: border-box;
    }
    .cam-card {
      position: relative;
      background: var(--card-bg-dark);
      border-radius: 16px;
      padding: 1rem;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      flex-direction: column;
      align-items: center;
      box-sizing: border-box;
      overflow: hidden;
      width: 612px;
      min-width: 0;
      min-height: 0;
      margin-bottom: 1.5rem;
      transition: box-shadow 0.3s, transform 0.3s, background 0.4s, border 0.3s;
      z-index: 1;
      min-height: 347px;
    }
    html[data-theme="light"] .cam-card {
      background: var(--card-bg-light);
      border: 1px solid #c9d6e3;
      color: #222;
    }
    .cam-card:hover {
      box-shadow: 0 16px 48px #00c6ff33, 0 2px 12px #0072ff22;
      transform: scale(1.025) translateY(-2px);
      z-index: 2;
    }
    .cam-title {
      font-size: 1.2rem;
      font-weight: 600;
      color: black;
      margin-bottom: 0.5rem;
      text-align: center;
      word-break: break-word;
      transition: color 0.4s;
    }
    html[data-theme="light"] .cam-title { color: #222; }
    .cam-live-img {
      width: 612px;
      height: 261px;
      max-width: 100vw;
      object-fit: cover;
      border-radius: 10px;
      border: 2px solid rgba(255, 255, 255, 0.15);
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
      background: #1b2838;
      aspect-ratio: 612 / 261;
      min-width: 0;
      min-height: 0;
      transition: width 0.25s, height 0.25s, box-shadow 0.3s;
      display: block;
      opacity: 1;
      z-index: 1;
    }
    .cam-card:hover .cam-live-img {
      box-shadow: 0 8px 32px #00c6ff99;
      opacity: 0.97;
    }
    .cam-loading,
    .cam-error {
      position: absolute;
      left: 0; 
      top: 44px;
      width: 612px;
      height: 261px;
      min-height: 261px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 2;
      transition: background 0.3s, border 0.3s;
    }
    .cam-loading {
      background: #222;
    }
    .cam-spinner {
      border: 6px solid #f3f3f3;
      border-top: 6px solid #00c6ff;
      border-radius: 50%;
      width: 48px;
      height: 48px;
      animation: spin 1s linear infinite;
      margin: 0 auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg);}
      100% { transform: rotate(360deg);}
    }
    .cam-error {
      background: var(--offline-bg);
      border: 3px dashed var(--offline-border);
      color: #ff4444;
      font-weight: 600;
      font-size: 1.12rem;
      text-align: center;
      padding: 0 1rem;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      gap: 0.8rem;
    }
    .cam-card.cam-offline {
      border: 3px solid var(--offline-border) !important;
      background: var(--offline-bg) !important;
      box-shadow: 0 0 8px 1px #ff444488 !important;
    }
    .cam-error-img {
      width: 80px;
      height: 80px;
      background: #ff4444;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 0.3rem auto;
      box-shadow: 0 2px 12px #ff444455, 0 0 0 4px #fff3;
    }
    .cam-error-img svg {
      width: 40px;
      height: 40px;
      fill: #fff;
      filter: drop-shadow(0 2px 4px #b00b);
    }
    .cam-footer {
      font-size: 0.9rem;
      color: #aaa;
      margin-top: 0.5rem;
      text-align: center;
      word-break: break-word;
      transition: color 0.4s;
    }
    html[data-theme="light"] .cam-footer { color: #444; }
    .live-indicator {
      position: absolute;
      left: 18px;
      top: 18px;
      display: flex;
      align-items: center;
      z-index: 3;
    }
    .live-dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #00ff47;
      box-shadow: 0 0 8px 2px #00ff47cc;
      margin-right: 8px;
      animation: dot-pulse 1.2s infinite alternate;
      border: 2px solid #222;
    }
    @keyframes dot-pulse {
      0% { box-shadow: 0 0 8px 1px #00ff47cc;}
      100% { box-shadow: 0 0 16px 4px #00ff47cc;}
    }
    .live-ind-txt {
      font-size: 0.95rem;
      color: #00ff47;
      font-weight: 500;
      text-shadow: 0 0 8px #222;
    }
    .history-section {
      padding: 1.5rem 0.5rem;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    h1 {
      font-size: 1.8rem;
      margin-bottom: 1rem;
      color: #ffffff;
      transition: color 0.4s;
    }
    html[data-theme="light"] h1 { color: #222; }
    table {
      width: 100%;
      max-width: 100%;
      border-collapse: collapse;
      background: var(--table-bg-dark);
      border-radius: 15px;
      overflow: hidden;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
      margin-bottom: 2rem;
      word-break: break-word;
      transition: background 0.4s;
    }
    html[data-theme="light"] table {
      background: var(--table-bg-light);
      color: #222;
    }
    th, td {
      padding: 0.75rem 1rem;
      text-align: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      word-break: break-word;
      transition: background 0.4s, color 0.4s;
    }
    th {
      background-color: rgba(0, 0, 0, 0.3);
      color: #ffffff;
      font-weight: 600;
    }
    html[data-theme="light"] th {
      background: var(--table-header-light);
      color: #333;
    }
    td {
      color: #f1f1f1;
    }
    html[data-theme="light"] td { color: #222; }
    .img-thumb {
      width: 70px;
      height: auto;
      border-radius: 8px;
      transition: transform 0.3s, box-shadow 0.3s;
      border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .img-thumb:hover {
      transform: scale(1.08) rotate(-1deg);
      box-shadow: 0 6px 32px #0072ff44;
      border: 2px solid #00c6ff;
    }
    .no-detections, .no-notifications {
      color: #bbb;
      font-style: italic;
    }
    footer {
      background: var(--footer-dark);
      color: #ccc;
      text-align: center;
      padding: 1rem 0;
      font-size: 0.9rem;
      margin-top: auto;
      border-radius: 12px;
      margin-top: 1.5rem;
      transition: background 0.4s, color 0.4s;
    }
    html[data-theme="light"] footer {
      background: var(--footer-light);
      color: #fff;
    }
    .footer-content {
      max-width: 960px;
      margin: 0 auto;
      padding: 0 1rem;
      line-height: 1.6;
    }
    .footer-content .rights {
      display: block;
      margin-top: 0.4rem;
      font-style: italic;
    }
    /* Responsive Design */
    @media (max-width: 1100px) {
      .cam-card, .cam-live-img, .cam-loading, .cam-error { width: 90vw; max-width: 612px;}
      .camera-grid { gap: 1rem; padding: 1rem; }
    }
    @media (max-width: 900px) {
      .cam-card, .cam-live-img, .cam-loading, .cam-error { width: 98vw !important; max-width: 650px;}
      .camera-grid { gap: 0.8rem; padding: 0.5rem; }
    }
    @media (max-width: 700px) {
      .cam-card, .cam-live-img, .cam-loading, .cam-error { width: 99vw !important; max-width: 100vw !important; }
      .cam-live-img { height: auto !important; }
      .camera-grid { gap: 0.7rem; padding: 0.1rem; }
    }
    @media (max-width: 576px) {
      body { padding: 0.1rem; }
      header {
        flex-direction: column;
        align-items: stretch;
        gap: 0.6rem;
        padding: 0.6rem 0.3rem;
        min-height: unset;
      }
      .header-section,
      .header-actions,
      .header-center {
        justify-content: center;
        width: 100%;
        flex: unset;
        min-width: 0;
        gap: 0.5rem;
      }
      .dashboard-title {
        font-size: 1.05rem;
        padding: 0.2rem 0;
        width: 100%;
      }
      .user-info {
        font-size: 0.8rem;
        position: static;
        text-align: center;
      }
      .controls button { padding: 0.4rem 0.8rem; font-size: 0.9rem; margin: 0.3rem; }
      .header-actions { gap: 0.5rem; }
      .toggle-label { font-size: 0.85rem; }
      .logout-link { font-size: 0.9rem; }
      .bell-icon { font-size: 1.2rem; padding: 0.4rem;}
      footer { font-size: 0.8rem; padding: 0.7rem 0; }
      .img-thumb { width: 50px; }
      th, td { font-size: 0.8rem; padding: 0.4rem; }
      h1 { font-size: 1.1rem; }
      .notification-content { max-width: 100vw; padding: 0.5rem; }
      .close-button { width: 25px; height: 25px; font-size: 1.2rem; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-section">
      <span class="user-info">Welcome, {{ session['username'] }}</span>
    </div>
    <div class="header-center">
      <span class="dashboard-title">Smart Surveillance Dashboard</span>
    </div>
    <div class="header-actions">
      <span class="toggle-theme-container">
        <span class="toggle-label" id="themeModeLabel">Dark Mode</span>
        <label class="theme-switch" title="Toggle light/dark mode">
          <input type="checkbox" id="themeToggler" />
          <span class="slider"></span>
        </label>
      </span>
      <span class="bell-icon" onclick="toggleNotifications()">🔔</span>
      <a href="{{ url_for('logout') }}" class="logout-link">Logout</a>
    </div>
  </header>
  <div class="notification-modal" id="notificationModal">
    <div class="notification-content">
      <span class="close-button" onclick="toggleNotifications()">×</span>
      <div class="history-section">
        <h1>Recent Notifications</h1>
        <table class="notification-table">
          <tr>
            <th>Image</th>
            <th>Timestamp</th>
            <th>Message</th>
          </tr>
          {% for notification in notifications %}
          <tr>
            <td>
              {% if notification.image %}
              <a href="{{ url_for('history_image', filename=notification.image) }}" target="_blank">
                <img src="{{ url_for('history_image', filename=notification.image) }}" class="img-thumb">
              </a>
              {% else %}
              <span>-</span>
              {% endif %}
            </td>
            <td>{{ notification.timestamp }}</td>
            <td>{{ notification.message }}</td>
          </tr>
          {% endfor %}
          {% if not notifications %}
          <tr><td colspan="3" class="no-notifications">No notifications yet</td></tr>
          {% endif %}
        </table>
        <h1>Detection History</h1>
        <table class="history-table">
          <tr>
            <th>Image</th>
            <th>Date & Time</th>
            <th>Person ID</th>
            <th>Camera</th>
          </tr>
          {% for det in detections %}
          <tr>
            <td>
              <a href="{{ url_for('history_image', filename=det.img) }}" target="_blank">
                <img src="{{ url_for('history_image', filename=det.img) }}" class="img-thumb">
              </a>
            </td>
            <td>{{ det.timestamp }}</td>
            <td>{{ det.person_id }}</td>
            <td>{{ det.camera }}</td>
          </tr>
          {% endfor %}
          {% if not detections %}
          <tr><td colspan="4" class="no-detections">No detections yet</td></tr>
          {% endif %}
        </table>
      </div>
    </div>
  </div>
  <div class="controls">
    <button onclick="setLayout(2)">2 Frames</button>
    <button onclick="setLayout(4)">4 Frames</button>
    <button onclick="setLayout(8)">8 Frames</button>
    <button onclick="setLayout(16)">16 Frames</button>
    </div>
  <div id="cameraGrid" class="camera-grid"></div>
  <footer>
    <div class="footer-content">
      © {{ year }} Primine Software Limited<br>
      201, Atharva Apartment, Besa - Manish Nagar Rd, Jai Hind Society, Nagpur, Maharashtra 440015<br>
      <span class="rights">All rights reserved.</span>
    </div>
  </footer>
  <script>
    // Theme toggler
    const themeToggler = document.getElementById('themeToggler');
    const themeLabel = document.getElementById('themeModeLabel');
    function applyTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem('theme', theme);
      themeToggler.checked = (theme === "light");
      themeLabel.textContent = theme === "light" ? "Light Mode" : "Dark Mode";
    }
    (function() {
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme) applyTheme(savedTheme);
      else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
        applyTheme("light");
      }
      themeToggler.addEventListener('change', () => {
        applyTheme(themeToggler.checked ? "light" : "dark");
      });
    })();

    const grid = document.getElementById('cameraGrid');
    const notificationModal = document.getElementById('notificationModal');
    const defaultStreams = [
      {% for cam in cams %}
      '{{ url_for("video_feed", cam_id=loop.index0) }}'{% if not loop.last %},{% endif %}
      {% endfor %}
    ];
    const camDetails = [
      {% for cam in cams %}
      { name: '{{ cam.name }}', footer: '{{ cam.footer }}', color: '{{ cam.color }}' }{% if not loop.last %},{% endif %}
      {% endfor %}
    ];

    // Camera SVG Loader template
    function getSpinner() {
      // Camera icon with animated lens (spinner)
      return `<div class="cam-loading">
        <svg width="64" height="48" viewBox="0 0 64 48" fill="none">
          <rect x="4" y="12" width="56" height="28" rx="6" fill="#222" stroke="#00c6ff" stroke-width="3"/>
          <rect x="22" y="4" width="20" height="10" rx="3" fill="#00c6ff" stroke="#00c6ff" stroke-width="1.5"/>
          <g>
            <circle cx="32" cy="26" r="9" fill="#111"/>
            <circle class="cam-lens-spinner" cx="32" cy="26" r="7" stroke="#00c6ff" stroke-width="3" fill="none" />
          </g>
        </svg>
        <style>
          .cam-lens-spinner {
            stroke-dasharray: 44;
            stroke-dashoffset: 0;
            animation: camspin 1s linear infinite;
            transform-origin: 32px 26px;
          }
          @keyframes camspin {
            0% { stroke-dashoffset: 44; transform: rotate(0deg);}
            100% { stroke-dashoffset: 0; transform: rotate(360deg);}
          }
        </style>
      </div>`;
    }

    // Error fallback template
    function getErrorFallback() {
      return `<div class="cam-error">
        <div class="cam-error-img">
          <!-- Camera with Slash SVG Icon -->
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <rect x="4" y="12" width="40" height="24" rx="4" stroke="#fff" stroke-width="3" fill="#ff4444"/>
            <circle cx="24" cy="24" r="7" stroke="#fff" stroke-width="3" fill="none"/>
            <rect x="16" y="6" width="16" height="6" rx="2" fill="#fff" stroke="#ff4444" stroke-width="2"/>
            <line x1="8" y1="40" x2="40" y2="8" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
          </svg>
        </div>
        <div>No Camera</div>
      </div>`;
    }

    // Live status indicator
    function getLiveIndicator() {
      return `<div class="live-indicator"><span class="live-dot"></span><span class="live-ind-txt">LIVE</span></div>`;
    }

    function setLayout(count) {
      grid.innerHTML = '';
      for (let i = 0; i < count; i++) {
        const div = document.createElement('div');
        div.className = 'cam-card';
        const src = i < defaultStreams.length ? defaultStreams[i] : `https://via.placeholder.com/640x360?text=Camera+${i+1}`;
        const name = i < camDetails.length ? camDetails[i].name : `Camera ${i+1}`;
        const footer = i < camDetails.length ? camDetails[i].footer : 'PRIMINE SOFTWARE PVT.LTD';
        const color = i < camDetails.length ? camDetails[i].color + '40' : '#ffffff40';
        div.style.backgroundColor = color;
        div.innerHTML = `
          ${getLiveIndicator()}
          <div class="cam-title">${name}</div>
          ${getSpinner()}
          <img class="cam-live-img" src="${src}" alt="Camera ${i + 1}" style="display:none" loading="lazy" />
          <div class="cam-footer">${footer}</div>
        `;
        grid.appendChild(div);

        const img = div.querySelector('.cam-live-img');
        const spinner = div.querySelector('.cam-loading');
        setTimeout(() => {
          img.style.display = "";
          spinner && spinner.remove();
        }, 1200 + Math.random()*600);

        img.onerror = () => {
          img.style.display = "none";
          spinner && spinner.remove();
          if (!div.querySelector('.cam-error')) {
            div.className += ' cam-offline';
            div.insertAdjacentHTML('beforeend', getErrorFallback());
            div.querySelector('.live-indicator').style.display = 'none';
          }
        }
        if (i >= defaultStreams.length && Math.random() < 0.33) {
          setTimeout(() => {
            img.onerror();
          }, 1600);
        }
      }
    }

    function toggleNotifications() {
      notificationModal.style.display = notificationModal.style.display === 'block' ? 'none' : 'block';
    }

    window.addEventListener('resize', () => {
      const camCount = grid.childElementCount || 2;
      setLayout(camCount);
    });

    window.onload = () => setLayout(2);
  </script>
</body>
</html>
    """, cams=CAMERA_LIST, detections=detections, notifications=notifications, year=datetime.now().year)

@app.route("/video_feed/<int:cam_id>")
@login_required
def video_feed(cam_id):
    return Response(gen_camera(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/history_img/<filename>")
@login_required
def history_image(filename):
    return send_from_directory(SAVE_DIR, filename)

# ===================== DETECTION + LIVE FEED ========================
def detect_and_alert():
    trackers = [SmartPersonTracker(max_disappeared=86400, max_history=40, match_radius=150) for _ in CAMERA_LIST]
    notified_persons = load_notified_persons()
    caps = [None for _ in CAMERA_LIST]
    reconnect_attempts = [0 for _ in CAMERA_LIST]
    max_reconnect_attempts = 5

    while True:
        try:
            for idx, cam in enumerate(CAMERA_LIST):
                if caps[idx] is None or not caps[idx].isOpened():
                    print(f"Cam {idx+1}: Attempting to connect to RTSP stream...")
                    caps[idx] = cv2.VideoCapture(cam["rtsp_url"])
                    if not caps[idx].isOpened():
                        reconnect_attempts[idx] += 1
                        if reconnect_attempts[idx] > max_reconnect_attempts:
                            print(f"Cam {idx+1}: Max reconnect attempts reached. Skipping this camera.")
                            caps[idx] = None
                            continue
                        print(f"Cam {idx+1}: Retry {reconnect_attempts[idx]}/{max_reconnect_attempts}")
                        time.sleep(3)
                        continue
                    reconnect_attempts[idx] = 0

                ret, frame = caps[idx].read()
                if not ret or frame is None:
                    print(f"Cam {idx+1}: Failed to read frame. Reconnecting...")
                    caps[idx].release()
                    caps[idx] = None
                    continue

                update_live_frames(idx, frame)

                height, width, _ = frame.shape
                # Updated YOLO input size to 416x416 for better accuracy with yolov2-tiny
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                yolo.setInput(blob)
                output_layers = yolo.getUnconnectedOutLayersNames()
                try:
                    outputs = yolo.forward(output_layers)
                    if outputs is None or len(outputs) == 0:
                        print(f"Cam {idx+1}: YOLO inference returned empty output. Skipping frame.")
                        continue
                except Exception as e:
                    print(f"Cam {idx+1}: YOLO inference failed: {e}. Skipping frame.")
                    continue

                boxes, confidences, class_ids = [], [], []
                human_detected = False

                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5 and classes[class_id] == "person":
                            human_detected = True
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_SIMPLEX

                rects = []
                if len(indexes) > 0 and human_detected:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        # Ensure bounding box coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        if w > 0 and h > 0:  # Only add valid bounding boxes
                            rects.append([x, y, w, h])

                objects = trackers[idx].update(rects)

                for person_id, (cX, cY, x, y, w, h) in objects.items():
                    label = f"Person ID: {person_id}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), font, 0.7, color, 2)
                    cv2.circle(frame, (cX, cY), 4, color, -1)

                    if str(person_id) not in notified_persons[idx]:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        display_ts = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                        filename = f"detection_{timestamp}_ID{person_id}_cam{idx+1}.jpg"
                        filepath = os.path.join(SAVE_DIR, filename)
                        notification = f"New Person (ID: {person_id}) detected [Camera {idx+1}]"
                        try:
                            os.makedirs(SAVE_DIR, exist_ok=True)
                            success = cv2.imwrite(filepath, frame)
                            if not success:
                                print(f"Cam {idx+1}: Failed to save image: {filepath}")
                                continue
                            if not os.path.exists(filepath):
                                print(f"Cam {idx+1}: Image not found after saving: {filepath}")
                                continue
                            send_telegram_message(f"⚠️ {notification} at {display_ts}")
                            send_telegram_photo(filepath, caption=f"Person ID: {person_id} at {display_ts} [Camera {idx+1}]")
                            log_notification(notification, filename)
                            notified_persons[idx].add(str(person_id))
                            save_notified_persons(notified_persons)
                            print(f"Cam {idx+1}: Notification logged and sent for Person ID: {person_id} at {display_ts}")
                        except Exception as e:
                            print(f"Cam {idx+1}: Error saving image or sending notification: {e}")
                    else:
                        # 👀 Repeated Visitor Detected
                        repeat_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        repeat_disp_ts = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                        repeat_filename = f"reentry_{repeat_ts}_ID{person_id}_cam{idx+1}.jpg"
                        repeat_filepath = os.path.join(SAVE_DIR, repeat_filename)
                        repeat_msg = f"👀 Repeat Visitor: Person ID {person_id} re-entered [Camera {idx+1}]"
                        try:
                            cv2.imwrite(repeat_filepath, frame)
                            send_telegram_message(f"{repeat_msg} at {repeat_disp_ts}")
                            send_telegram_photo(repeat_filepath, caption=f"Re-entry: Person ID {person_id} at {repeat_disp_ts} [Camera {idx+1}]")
                            log_notification(repeat_msg, repeat_filename)
                            print(f"Cam {idx+1}: Re-entry alert sent for Person ID: {person_id} at {repeat_disp_ts}")
                        except Exception as e:
                            print(f"Cam {idx+1}: Error during repeat visitor alert: {e}")

                # if idx == 0:
                #     cv2.imshow("YOLO Detection", frame)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         print("Received 'q' key, initiating shutdown...")
                #         for cap in caps:
                #             if cap:
                #                 cap.release()
                #         cv2.destroyAllWindows()
                #         sys.exit(0)

            time.sleep(0.01)

        except Exception as e:
            print(f"Error in detect_and_alert loop: {e}")
            time.sleep(1)
            continue

 
# ===================== RUN ========================
if __name__ == "__main__":
    # Start detection in a background thread
    detection_thread = threading.Thread(target=detect_and_alert, daemon=True)
    detection_thread.start()
    # Run Flask in the main thread (so login page and dashboard work)
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)



