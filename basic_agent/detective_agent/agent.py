import firebase_admin
from firebase_admin import credentials, firestore
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio


# --- Initialize Firebase ---
cred = credentials.Certificate("/Users/iaddchehaeb/Documents/GitHub/KHVIII-Personal_Bubble/service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# --- Define Detective Agent ---
# detective_agent = Agent(
#     name="detective_agent",
#     model="gemini-2.0-flash-exp",
#     description="Determines whether a person has suspicious activity by the amount of times they have appeared on camera and the amount of time they have been on camera.",
#     instruction="""
#     You are a detective agent that makes the decision on whether someone is suspicious based on the data you receive. A suspicious person will have a lot of time on camera (20 seconds), and various reappearances (3 or more). Always provide a list of people that may be suspicious if there are any. 
#     If you decide the person is suspicious format your response like this 'id: suspicious' or 'id: not suspicious'
#     """,
# )

detective_agent = LlmAgent(
    name="detective_agent",
    model="models/gemini-2.5-flash",
    instruction="You are a detective agent that analyzes detected person data..."
)

# --- Create Runner with app_name ---
runner = InMemoryRunner(
    agent=detective_agent,
    app_name="agents"  # Only the runner needs app_name
)


def on_snapshot(col_snapshot, changes, read_time):
    combined_people = []

    for change in changes:
        if change.type.name == "ADDED":
            new_data = change.document.to_dict()
            print("🆕 New detection data received:", new_data)
            combined_people.extend(new_data.get("people", []))

    if combined_people:
        combined_data = {"people": combined_people}
        analyzed = analyze_people_data(combined_data)
        print(f"\n📊 Analyzed data being sent to agent:\n{analyzed}\n")

        async def call_agent():
            try:
                response_text = ""
                content = types.Content(
                    role='user',
                    parts=[types.Part(text=analyzed)]
                )
                
                print("🔄 Starting agent execution...")
                
                # Process all events from the agent
                async for event in runner.run_async(
                    user_id="security_system",
                    session_id="main_session",
                    new_message=content
                ):
                    print(f"📥 Received event: {type(event).__name__}")
                    
                    # Check different event types
                    if hasattr(event, 'content') and event.content:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_chunk = part.text
                                response_text += text_chunk
                                print(f"🔎 Agent says: {text_chunk}", flush=True)
                
                if not response_text:
                    print("⚠️ Warning: No response text received from agent")
                
                return response_text
            
            except Exception as e:
                print(f"❌ Error in call_agent: {e}")
                import traceback
                traceback.print_exc()
                return None

        response = asyncio.run(call_agent())
        
        if response:
            print(f"\n✅ Detective Agent Full Response:\n{response}\n")
        else:
            print("\n⚠️ No response from agent\n")


def analyze_people_data(data):
    """
    Example data analysis logic.
    'data' could be a dict with people detections, timestamps, etc.
    """
    result = []
    for person in data.get("people", []):
        result.append(
            f"Person {person['id']} reappeared {person['reappearances']} times "
            f"and spent {person['timeSpent']} seconds on screen."
        )
    return "\n".join(result)


# --- Start Firestore Listener ---
col_query = db.collection("entities")
query_watch = col_query.on_snapshot(on_snapshot)

print("👂 Listening for new camera detections...")
while True:
    pass  # Keeps the listener running
