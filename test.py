from google.genai import Client
import os
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))
chat = client.chats.create(model="gemini-2.5-flash")
resp = chat.send_message("Say 'OK' only.")
print(resp.text)
