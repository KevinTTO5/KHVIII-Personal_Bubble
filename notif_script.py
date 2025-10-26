# send_test_fcm.py
import os
import firebase_admin
from firebase_admin import credentials, messaging

# ---- 1) init Admin SDK exactly once ----
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")

if not firebase_admin._apps:  # prevents "already initialized" if you rerun in REPL
    cred = credentials.Certificate(KEY_PATH)
    app = firebase_admin.initialize_app(cred)
else:
    app = firebase_admin.get_app()

# ---- 2) choose one: topic or device token ----
TOPIC = "user_1"                 # if your app subscribed to "user_1"
DEVICE_TOKEN = None              # or paste a real FCM registration token string

# ---- 3) build a high-priority DATA message (your service expects this) ----
msg = messaging.Message(
    data={
        "type": "ALERT",
        "msg": "Cyclops: high alert — person behind you for 18s!"
    },
    android=messaging.AndroidConfig(priority="high"),
    topic=TOPIC if not DEVICE_TOKEN else None,
    token=DEVICE_TOKEN if DEVICE_TOKEN else None,
)

# ---- 4) send ----
resp = messaging.send(msg, app=app)
print("✅ Sent. Message ID:", resp)
