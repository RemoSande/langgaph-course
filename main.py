from dotenv import load_dotenv
load_dotenv()
from pprint import pprint

from graph.graph import app

question1 = "What are the types of agent memory?"
client_topics = ['agents', 'prompt engineering', 'adversarial attacks']

inputs = {
    "client_topics": client_topics,
    "question": question1
}

for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
        pprint(value.get("generation", "No generation available"))