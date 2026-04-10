import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def start_chat():
    print("--- Groq Chat Initialized ---")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Chat ended.")
            break

        try:
            # Simple request to the LLM
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": user_input}],
                temperature=0.7,
            )

            # Print just the message content
            print(f"Groq: {completion.choices[0].message.content}\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    start_chat()
