import firebase_admin
from firebase_admin import credentials, firestore
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

import os
import traceback
import asyncio
import time
from dotenv import load_dotenv, find_dotenv

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# Load .env if present
load_dotenv(find_dotenv())
# --- Configure Google GenAI API key if available ---
_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
if _api_key:
    print("🔑 API key loaded successfully")
else:
    print("⚠️ API key not found")

# --- Initialize Firebase ---
cred = credentials.Certificate("/Users/williambu/hackathon-25/KHVIII-Personal_Bubble/service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
root_agent = LlmAgent(
    name="detective_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are a detective agent that identifies suspicious people by detecting cumulative patterns across the entire session.

    Rules:
    - Maintain running totals per person id across all prior messages in this session.
    - For each id, accumulate: total_time_on_camera_seconds and total_reappearances.
    - Mark a person as suspicious if total_time_on_camera_seconds ≥ 20 OR total_reappearances ≥ 3.

    Output:
    - For every person id present in the latest input, output exactly one line with a brief reason: 'id: suspicious - because <reason>' or 'id: not suspicious - because <reason>'.
      The reason must reference the cumulative totals and the rule that applied, e.g., 'time=23.4s ≥ 20s' or 'reappearances=4 ≥ 3', or 'time=12.0s and reappearances=2 < thresholds'.
    - If there are any suspicious people, also include a final line 'suspicious_ids: <comma-separated ids>'.
    """
)

# --- Create Runner with app_name ---
runner = InMemoryRunner(
    agent=root_agent,
    app_name="agents"  # Only the runner needs app_name
)
# Ensure session exists for this runner/user/session_id
try:
    runner.session_service.create_session(
        app_name="agents", user_id="security_system", session_id="main_session"
    )
except Exception:
    pass
# --- Define Detective Agent ---
# root_agent = Agent(
#     name="detective_agent",
#     model="gemini-2.0-flash",
#     description="Receives new people detection data and deciphers it.",
#     instruction="""
#     You are a detective agent that makes the decision on whether someone is suspicious based on the data you receive. A suspicious person will have a lot of time on camera (20 seconds), and various reappearances (3 or more). Always provide a list of people that may be suspicious if there are any. 
#     If you decide the person is suspicious format your response like this 'id: suspicious' or 'id: not suspicious'
#     """,
# )

def on_snapshot(col_snapshot, changes, read_time):
    combined_people = []

    for change in changes:
        print(f"type is {change.type.name}")
        if change.type.name == "ADDED":
            new_data = change.document.to_dict()
            print("🆕 New detection data received:", new_data)

            # Each presence_window document has a "people_data" list of dicts
            people_list = new_data.get("people_data", [])
            for person in people_list:
                # Normalize optional fields safely
                combined_people.append({
                    "id": person.get("id"),
                    "reappearances": person.get("reappearance_counter", 0),
                    "timeSpent": person.get("time", 0.0),
                    "interval_id": new_data.get("interval_id"),
                    "start_time": new_data.get("start_time"),
                    "end_time": new_data.get("end_time")
                })

            print(f"Line successfully extended → total {len(combined_people)}")

    # only analyze once per batch
    if combined_people:
        combined_data = {"people": combined_people}
        analyzed = analyze_people_data(combined_data)
        print(f"\n📊 Analyzed data being sent to agent:\n{analyzed}\n")
        asyncio.run_coroutine_threadsafe(call_agent(analyzed), loop)



async def call_agent(analyzed):
    response_text = ""
    print("🔄 Starting agent execution...")
    try:
        # Build Content as expected by the runner
        content = types.Content(role="user", parts=[types.Part(text=analyzed)])
        async for event in runner.run_async(
            user_id="security_system",
            session_id="main_session",
            new_message=content
        ):
            print(f"📥 Received event: {type(event).__name__}")
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_chunk = part.text
                        response_text += text_chunk
                        print(f"🔎 Agent says: {text_chunk}", flush=True)
    except Exception as exc:
        print(f"❌ Agent run failed: {exc}")
        traceback.print_exc()
        return ""

    if not response_text:
        print("⚠️ Warning: No response text received from agent")
    return response_text


def analyze_people_data(data):
    """
    Each entry now uses fields:
      id, reappearances, timeSpent, interval_id, start_time, end_time
    """
    result = []
    for person in data.get("people", []):
        result.append(
            f"Person {person.get('id')} reappeared {person.get('reappearances')} times "
            f"and spent {person.get('timeSpent')} seconds on screen "
            f"during interval {person.get('interval_id')} "
            f"({person.get('start_time')} → {person.get('end_time')})."
        )
    return "\n".join(result)


async def start_firestore_listener():
    col_query = db.collection("presence_windows")  # changed from "entities"
    col_query.on_snapshot(on_snapshot)

    print("👂 Listening for new presence window data...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down listener...")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(create_llms)
    loop.run_until_complete(start_firestore_listener())
