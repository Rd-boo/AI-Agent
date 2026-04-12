import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def start_chat():
    print("--- Groq Chat Initialized ---")

    # Initialize conversation history
    memory_history = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Chat ended.")
            break

        try:
            # Add the user's message to the history
            memory_history.append({"role": "user", "content": user_input})

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=memory_history,
                temperature=0.7,
            )
            print(f"Groq: {completion.choices[0].message.content}\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    start_chat()
