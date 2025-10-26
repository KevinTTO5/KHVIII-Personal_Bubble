# ultrasonic_listener.py
import serial, time
from tools_alert import send_alert_to_topic

DEVICE_TOPIC = "user_1"
PORT = "/dev/tty.usbmodem12345"   # adjust this for your Arduino
THRESHOLD_M = 2.0
COOLDOWN_S = 8

def main():
    ser = serial.Serial(PORT, 9600, timeout=1)
    last_alert = 0

    while True:
        line = ser.readline().decode("utf-8").strip()
        if not line:
            continue
        try:
            distance = float(line)
        except ValueError:
            continue

        now = time.time()
        if distance <= THRESHOLD_M and now - last_alert > COOLDOWN_S:
            msg = f"Object detected within {distance:.2f} meters!"
            print("🚨 Ultrasonic Trigger:", msg)
            send_alert_to_topic(DEVICE_TOPIC, msg, severity="CRITICAL")
            last_alert = now

        time.sleep(0.1)

if __name__ == "__main__":
    main()
