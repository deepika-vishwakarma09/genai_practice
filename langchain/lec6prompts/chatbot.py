
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()  # Loads your Hugging Face token from .env

# Initialize Hugging Face model (use the one you like, e.g., Falcon, Flan-T5, TinyLlama)
model = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation" # "text2text-generation" for T5-like models
)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    chat_history.append(HumanMessage(content=user_input))

    # Prepare plain text prompt (Hugging Face models don't use message objects)
    history_text = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage)
        else f"AI: {msg.content}" if isinstance(msg, AIMessage)
        else f"{msg.content}"  # System message
        for msg in chat_history
    ])

    # Get model response
    result = model.invoke(history_text)
    ai_reply = result.strip()

    chat_history.append(AIMessage(content=ai_reply))
    print("AI:", ai_reply)

# Final history display
print("\n--- Full Chat History ---")
for msg in chat_history:
    print(f"{msg.__class__.__name__}: {msg.content}")
