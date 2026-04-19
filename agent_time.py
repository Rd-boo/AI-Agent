import os
import json
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def real_time():
    date = datetime.now()   # Get the current date and time
    return date.strftime("%H:%M %A, %B %d %Y")  # Hour:Minute Day_name, Month_name Day_number Year


def content() -> str:
    return """
    You're an agent of the real time.
    You have access to a tool called 'real_time' that gives you the current real time.
    The format of the time must be:
        Today's date: HH:MM [PM||AM] day_name month_name day_number year
    For example:
        Today's date: 14:30 AM Tuesday September 5 2023
    If the user ask about another time, then always base your answer on the current real time, no need to explain the answer.
    Otherswise, you don't have access to any other information about the real time.
    """


def run_agent():
    # Initialize conversation history
    messages = [{"role": "system", "content": content()}]

    while True:

        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "stop"]:
            break   # Quit the chat

        messages.append({"role": "user", "content": user_input})

        # Describe the tool to the LLM (The "Manual")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "real_time",
                    "description": "Get the current real time",
                    "parameters": {
                        "type": "object",
                        "properties": {},   # No parameters needed for this function
                    },
                },
            }
        ]

        # LLM decides to use the tool
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message  # The LLM's response message
        tool_calls = response_message.tool_calls    # Check if the LLM decided to call any tools

        if tool_calls:  # LLM decided to call a tool

            messages.append(response_message)   # Add tool call message to history
            function_name = tool_calls[0].function.name # Get the name of the tool function
            function_response = real_time() # Call the actual function to get the real time

            print("=" * 30)
            print(f"[Tool Call] {function_name}() ...")
            print("=" * 30)

            # Add the RESULT back to history with the role "tool"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].id,
                    "name": function_name,
                    "content": str(function_response),
                }
            )

            # Final call: The LLM reads the result and gives a natural answer
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=messages
            )
            print(f"{final_response.choices[0].message.content}")

        else:
            # If the LLM didn't call a tool, just print its response
            print("*"*40)
            print("LLM response: No tool calls detected.")
            print("*"*40)
            print("Can't help you with that, I only have access to the real time.")


if __name__ == "__main__":
    run_agent()
