# test_adk_session.py
import os, asyncio
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService

assert os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), "No model credentials"

agent  = LlmAgent(name="probe", model=os.getenv("CYCLOPS_MODEL","gemini-2.5-flash"),
                  instruction="Reply with 'OK' once.")
runner = Runner(agent=agent, app_name="agents", session_service=InMemorySessionService())
SESSION_ID = "probe_session"

async def ensure_session():
    svc = getattr(runner, "session_service", None)
    create_fn = getattr(svc, "create_session", None) or getattr(svc, "start_session", None)
    assert create_fn, "No create_session/start_session on this ADK build"
    await create_fn(app_name=runner.app_name, user_id="u", session_id=SESSION_ID)

async def go():
    await ensure_session()
    content = types.Content(role="user", parts=[types.Part(text="ping")])
    async for ev in runner.run_async(user_id="u", session_id=SESSION_ID, new_message=content):
        if getattr(ev, "content", None) and ev.content.parts:
            print(ev.content.parts[0].text.strip())

asyncio.run(go())
