from endercom import AgentOptions, ServerOptions, create_server_agent
from mangum import Mangum
import os
from dotenv import load_dotenv

load_dotenv()

agent_options = AgentOptions(
    frequency_api_key=os.environ["FREQUENCY_API_KEY"],
    frequency_id=os.environ["FREQUENCY_ID"],
    agent_id=os.environ.get("AGENT_ID", "{{AGENT_NAME}}"),
)

server_options = ServerOptions(
    host="0.0.0.0",
    port=8004,
    enable_heartbeat=True,
    enable_a2a=True,
)

async def handle_message(message):
    return "Hi. I'm {{AGENT_NAME}}. Your original message was: " + message.content

agent = create_server_agent(agent_options, server_options)
agent.set_message_handler(handle_message)

app = agent.create_server_wrapper(server_options)
handler = Mangum(app)

# If you want to test the agent locally, uncomment the following line:
# agent.run()

# Then run:
# python app.py
# Then you can send a message to the agent at http://localhost:8004