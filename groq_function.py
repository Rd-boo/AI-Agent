import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 1. Addition function
def add_numbers(a, b):
    return a + b


# 2. Subtraction function
def subtract_numbers(a, b):
    return a - b


# 3. Multiply function
def multiply_numbers(a, b):
    return a * b


# 4. Division function
def divide_numbers(a, b):
    if b == 0:
        print(f"\n[SYSTEM]: Error! Cannot divide by zero.")
    return a / b


# "Phone book" of available functions
available_functions = {
    "add_numbers": add_numbers,
    "subtract_numbers": subtract_numbers,
    "multiply_numbers": multiply_numbers,
    "divide_numbers": divide_numbers,
}


def run_agent():

    # 1. Initialize conversation history
    messages = [
        {
            "role": "system",
            "content": "You are a helpful agent that can perform basic math operations using tools.",
        }
    ]

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Exiting.")
            break

        messages.append({"role": "user", "content": user_input})

        # 2. Describe the tool to the LLM (The "Manual")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "subtract_numbers",
                    "description": "Subtract two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply_numbers",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "divide_numbers",
                    "description": "Divide two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            },
        ]

        # 3. First call: The LLM decides to use the tool
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:  # If the LLM decided to call a tool
            messages.append(response_message)  # Add AI's request to history

            for tool_call in tool_calls:

                function_name = tool_call.function.name
                function_to_call = available_functions[
                    function_name
                ]  # Look up in our "phone book"
                function_args = json.loads(tool_call.function.arguments)

                # Execute the function dynamically
                function_response = function_to_call(
                    a=function_args.get("a"), b=function_args.get("b")
                )

                # 5. Add the RESULT back to history with the role "tool"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(function_response),
                    }
                )

            # 6. Final call: The LLM reads the result and gives a natural answer
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=messages
            )
            print(f"\nAgent: {final_response.choices[0].message.content}")

        else:
            # If the LLM didn't call a tool, just print its response
            print(f"\nAgent: {response_message.content}")


if __name__ == "__main__":
    run_agent()
