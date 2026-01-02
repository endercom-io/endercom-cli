# To run this agent locally, execute: python app.py

async def handle_message(message):
    return "Hi. I'm {{AGENT_NAME}}. Your original message was: " + message.content

