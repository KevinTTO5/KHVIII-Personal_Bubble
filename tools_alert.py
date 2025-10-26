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
DEVICE_TOKEN = None              # or paste a real FCM registration token string

def send_alert_to_topic(topic: str, message: str, severity: str = "MEDIUM", cooldown_s: int = 5):
    msg = messaging.Message(
        data={
            "type": "ALERT",
            "msg": message
        },
        android=messaging.AndroidConfig(priority="high"),
        topic=topic if not DEVICE_TOKEN else None,
        token=DEVICE_TOKEN if DEVICE_TOKEN else None,
    )
    response = messaging.send(msg, app=app)
    return response