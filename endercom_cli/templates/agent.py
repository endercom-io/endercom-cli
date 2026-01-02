# To run this agent locally, execute: python app.py

async def handle_message(message):
    # Write your agent logic here.
    # message.content contains the incoming message.

    # Simply return the response by calling the return statement.
    return "Hi. I'm {{AGENT_NAME}}. Your original message was: " + message.content

