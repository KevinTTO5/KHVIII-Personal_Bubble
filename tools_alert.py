# tools_alert.py
import firebase_admin
from firebase_admin import messaging

def send_alert_to_topic(topic: str, msg: str, severity: str = "MEDIUM", cooldown_s: int = 5):
    message = messaging.Message(
        notification=messaging.Notification(
            title=f"{severity} Alert!",
            body=msg,
        ),
        topic=topic,
    )
    response = messaging.send(message)
    return response
