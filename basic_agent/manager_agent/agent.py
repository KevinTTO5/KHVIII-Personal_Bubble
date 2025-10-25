from google.adk.agents import Agent
from google.adk.agents import LlmAgent


detective_agent = Agent(
    name="detective_agent",
    description="Determines whether a person has suspicious activity by the amount of times they have appeared on camera and the amount of time they have been on camera.",
    instruction="""
    You are a detective agent that makes the decision on whether someone is suspicious based on the data you receive. A suspicious person will have a lot of time on camera (20 seconds), and various reappearances (3 or more). Always provide a list of people that may be suspicious if there are any.
    """
)

video_interpreter_agent = Agent(
    name="video_interpreter_agent",
    description="Determine amount of reappearances a person made based on the time stamps it was on screen",
    instruction="""
    Follow these instructions:
Input: A JSON file containing multiple people detected on camera. Each person has a unique id and timestamps (timeStart, timeEnd).
Task:
Analyze the JSON to determine, for each unique id:
How many times the person reappears on camera (number of separate appearances).
The total time the person spent on screen (sum of all timeSpent).
Output:
Provide a clean, readable list showing, for each person:
Person <id>: <reappearances> reappearances, <total_time> seconds on screen
Do not return the original JSON.
Next step:
Immediately call detective_agent and send it the list of analyzed data.
    """,
    sub_agents=[detective_agent],
)

root_agent = Agent(
    name="manager_agent",
    # https://ai.google.dev/gemini-api/docs/models
    model="gemini-2.0-flash",
    description="Receives new person detection data as a JSON",
    instruction="""
    Receives new person detection data as a JSON. 
    Instantly gives that information to the Video-Interpreter agent by triggering it. 
    """,
    sub_agents = [video_interpreter_agent]
)