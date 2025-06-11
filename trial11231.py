import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import time
import requests
from collections import deque
from scipy.spatial import distance as dist
import threading
import json
import logging
from flask import Flask, render_template_string, Response, send_from_directory, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import signal
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Signal Handler for Ctrl+C
def signal_handler(sig, frame):
    logger.info("Received Ctrl+C, initiating shutdown...")
    raise SystemExit

signal.signal(signal.SIGINT, signal_handler)

# Telegram Setup
TELEGRAM_BOT_TOKEN = "7959551214:AAF4pUbQuItuttpMekUTxdD3EaivE41SDMI"
TELEGRAM_CHAT_ID = "6602085152"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram message sent successfully: %s", message)
    except requests.RequestException as e:
        logger.error("Failed to send Telegram message: %s", e)

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data, timeout=10)
            response.raise_for_status()
            logger.info("Telegram photo sent successfully: %s", photo_path)
    except requests.RequestException as e:
        logger.error("Failed to send Telegram photo: %s", e)
    except FileNotFoundError:
        logger.error("Photo file not found: %s", photo_path)

# YOLO Setup
try:
    yolo = cv2.dnn.readNet("yolov2-tiny.weights", "yolov2-tiny.cfg")
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
except Exception as e:
    logger.error("Error loading YOLO model: %s", e)
    sys.exit(1)

try:
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()
except Exception as e:
    logger.error("Error loading COCO names: %s", e)
    sys.exit(1)

SAVE_DIR = "detections"
NOTIFICATION_LOG = "notifications.txt"
NOTIFIED_PERSONS_FILE = "notified_persons.json"
USERS_FILE = "users.json"
os.makedirs(SAVE_DIR, exist_ok=True)

# User Management
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error("Error loading users: %s", e)
        return {}

def save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error("Error saving users: %s", e)

# Write Notification to Text File
def log_notification(message, image_filename):
    try:
        with open(NOTIFICATION_LOG, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message} | {image_filename}\n")
    except Exception as e:
        logger.error("Failed to log notification: %s", e)

# Load/Save Notified Persons
def load_notified_persons():
    try:
        if os.path.exists(NOTIFIED_PERSONS_FILE):
            with open(NOTIFIED_PERSONS_FILE, "r") as f:
                data = json.load(f)
                return [set(data.get(f"cam_{i+1}", [])) for i in range(len(CAMERA_LIST))]
        return [set() for _ in CAMERA_LIST]
    except Exception as e:
        logger.error("Error loading notified persons: %s", e)
        return [set() for _ in CAMERA_LIST]

def save_notified_persons(notified_persons):
    try:
        data = {f"cam_{i+1}": list(notified_persons[i]) for i in range(len(CAMERA_LIST))}
        with open(NOTIFIED_PERSONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error("Error saving notified persons: %s", e)

# Modified SmartPersonTracker for Railway Station
class SmartPersonTracker:
    def __init__(self, max_disappeared=300, max_history=40, match_radius=200):
        self.person_id = 1
        self.persons = []
        self.max_disappeared = max_disappeared  # 300 seconds for disappearance
        self.max_history = max_history
        self.match_radius = match_radius  # Increased for crowded railway station
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
            logger.error("Error loading last person ID: %s", e)

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
            min_distance = float('inf')
            best_match = None

            # Check all existing persons for a match
            for person in self.persons:
                # Skip if person has disappeared for too long
                if (datetime.now() - person['last_seen']).total_seconds() > self.max_disappeared:
                    continue

                prev_cX, prev_cY = person['trace'][-1] if person['trace'] else person['centroid']
                distance = dist.euclidean([cX, cY], [prev_cX, prev_cY])
                if distance < self.match_radius and distance < min_distance:
                    min_distance = distance
                    best_match = person

            if best_match:
                # Update existing person
                best_match['centroid'] = (cX, cY)
                best_match['trace'].append((cX, cY))
                if len(best_match['trace']) > self.max_history:
                    best_match['trace'].popleft()
                best_match['last_seen'] = datetime.now()
                found = True
                matched.append(best_match['id'])
                objects_out[best_match['id']] = (cX, cY, x, y, w, h)

            if not found:
                # Create new person only if no match is found
                new_person = {
                    'id': self.person_id,
                    'centroid': (cX, cY),
                    'trace': deque([(cX, cY)], maxlen=self.max_history),
                    'last_seen': datetime.now(),
                    'first_seen': datetime.now()  # Track when person was first detected
                }
                self.persons.append(new_person)
                objects_out[self.person_id] = (cX, cY, x, y, w, h)
                matched.append(self.person_id)
                self.person_id += 1

        # Retain persons who haven't disappeared for too long
        self.persons = [p for p in self.persons if (datetime.now() - p['last_seen']).total_seconds() < self.max_disappeared]
        return objects_out

# Camera Configuration for Railway Station
CAMERA_LIST = [
    {
        "name": "Platform 1 Surveillance",
        "footer": "Railway Security System",
        "color": "#f7cac9",
        "rtsp_url": "rtsp://admin:ADMIN%40123@192.168.29.163:554/cam/realmonitor?channel=1&subtype=1"  # Use H.264 substream
    },
    {
        "name": "Main Entrance Surveillance",
        "footer": "Railway Security System",
        "color": "#dbe6e4",
        "rtsp_url": "rtsp://admin:ADMIN%40123@192.168.29.163:554/cam/realmonitor?channel=2&subtype=1"
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

# Flask Setup with Authentication (unchanged, but included for completeness)
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash("Please log in to access the railway surveillance dashboard.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Register and Login routes remain unchanged (omitted for brevity)

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
                "camera": CAMERA_LIST[int(camera_id)-1]["name"]
            })
        except Exception as e:
            logger.error("Error parsing detection file %s: %s", fname, e)
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
                        logger.error("Error parsing notification: %s, %s", line, e)
                        continue
    except Exception as e:
        logger.error("Error reading notifications: %s", e)

    # Update dashboard template for railway context
    return render_template_string("""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Railway Surveillance Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    /* Same CSS as original, with minor updates for railway context */
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
    /* Rest of CSS remains unchanged */
  </style>
</head>
<body>
  <header>
    <div class="header-section">
      <span class="user-info">Welcome, {{ session['username'] }}</span>
    </div>
    <div class="header-center">
      <span class="dashboard-title">Railway Surveillance Dashboard</span>
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
            <th>Location</th>
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
      © {{ year }} Railway Security System<br>
      Managed by Railway Protection Authority<br>
      <span class="rights">All rights reserved.</span>
    </div>
  </footer>
  <script>
    // JavaScript remains unchanged
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

# Detection and Alert Function
def detect_and_alert():
    trackers = [SmartPersonTracker(max_disappeared=300, max_history=40, match_radius=200) for _ in CAMERA_LIST]
    notified_persons = load_notified_persons()
    caps = [None for _ in CAMERA_LIST]
    reconnect_attempts = [0 for _ in CAMERA_LIST]
    max_reconnect_attempts = 5
    frame_skip = 5
    frame_counts = [0 for _ in CAMERA_LIST]
    last_notification_time = [None for _ in CAMERA_LIST]
    notification_cooldown = 30  # seconds

    # Force TCP for RTSP to reduce packet loss
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    while True:
        try:
            for idx, cam in enumerate(CAMERA_LIST):
                if caps[idx] is None or not caps[idx].isOpened():
                    logger.info("Cam %d (%s): Attempting to connect to RTSP stream %s", idx+1, cam["name"], cam["rtsp_url"])
                    caps[idx] = cv2.VideoCapture(cam["rtsp_url"], cv2.CAP_FFMPEG)
                    caps[idx].set(cv2.CAP_PROP_BUFFERSIZE, 10)
                    caps[idx].set(cv2.CAP_PROP_FPS, 15)
                    if not caps[idx].isOpened():
                        reconnect_attempts[idx] += 1
                        if reconnect_attempts[idx] > max_reconnect_attempts:
                            logger.error("Cam %d (%s): Max reconnect attempts reached. Skipping this camera.", idx+1, cam["name"])
                            caps[idx] = None
                            update_live_frames(idx, None)
                            continue
                        logger.warning("Cam %d (%s): Retry %d/%d", idx+1, cam["name"], reconnect_attempts[idx], max_reconnect_attempts)
                        time.sleep(3)
                        continue
                    reconnect_attempts[idx] = 0

                ret, frame = caps[idx].read()
                if not ret or frame is None:
                    logger.error("Cam %d (%s): Failed to read frame. Reconnecting...", idx+1, cam["name"])
                    caps[idx].release()
                    caps[idx] = None
                    update_live_frames(idx, None)
                    continue

                update_live_frames(idx, frame.copy())
                frame_counts[idx] += 1
                if frame_counts[idx] % frame_skip != 0:
                    continue

                height, width, _ = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                yolo.setInput(blob)
                output_layers = yolo.getUnconnectedOutLayersNames()

                try:
                    outputs = yolo.forward(output_layers)
                    if outputs is None or len(outputs) == 0:
                        logger.warning("Cam %d (%s): YOLO inference returned empty output. Skipping frame.", idx+1, cam["name"])
                        continue
                except Exception as e:
                    logger.error("Cam %d (%s): YOLO inference failed: %s. Skipping frame.", idx+1, cam["name"], e)
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
                            x = max(0, int(center_x - w / 2))
                            y = max(0, int(center_y - h / 2))
                            w = min(w, width - x)
                            h = min(h, height - y)
                            if w > 0 and h > 0:
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                rects = [boxes[i] for i in indexes.flatten()] if len(indexes) > 0 else []

                objects = trackers[idx].update(rects)
                for person_id, (cX, cY, x, y, w, h) in objects.items():
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Person ID: {person_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(frame, (cX, cY), 4, color, -1)

                    # Check if notification should be sent (with cooldown)
                    if str(person_id) not in notified_persons[idx]:
                        person = next((p for p in trackers[idx].persons if p['id'] == person_id), None)
                        if person and (last_notification_time[idx] is None or 
                                      (datetime.now() - last_notification_time[idx]).total_seconds() > notification_cooldown):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            display_ts = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                            filename = f"detection_{timestamp}_ID{person_id}_cam{idx+1}.jpg"
                            filepath = os.path.join(SAVE_DIR, filename)
                            notification = f"New Person (ID: {person_id}) detected at {cam['name']}"
                            try:
                                os.makedirs(SAVE_DIR, exist_ok=True)
                                if not cv2.imwrite(filepath, frame):
                                    logger.error("Cam %d (%s): Failed to save image: %s", idx+1, cam["name"], filepath)
                                    continue
                                if not os.path.exists(filepath):
                                    logger.error("Cam %d (%s): Image not found after saving: %s", idx+1, cam["name"], filepath)
                                    continue
                                send_telegram_message(f"🚨 {notification} at {display_ts}")
                                send_telegram_photo(filepath, caption=f"Person ID: {person_id} at {display_ts} [{cam['name']}]")
                                log_notification(notification, filename)
                                notified_persons[idx].add(str(person_id))
                                save_notified_persons(notified_persons)
                                last_notification_time[idx] = datetime.now()
                                logger.info("Cam %d (%s): Notification sent for Person ID: %d at %s", idx+1, cam["name"], person_id, display_ts)
                            except Exception as e:
                                logger.error("Cam %d (%s): Error processing notification for Person ID: %d: %s", idx+1, cam["name"], person_id, e)

                update_live_frames(idx, frame)
            time.sleep(0.01)
        except Exception as e:
            logger.error("Error in detect_and_alert loop: %s", e)
            time.sleep(1)
            continue

if __name__ == "__main__":
    detection_thread = threading.Thread(target=detect_and_alert, daemon=True)
    detection_thread.start()
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)