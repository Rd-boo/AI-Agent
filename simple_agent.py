import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def instraction() -> str:
    return ("""
    You are a helpful agent, answer the users with the best u can based on your knowledge.
    Use the provided tools *prayer_times* and *weather_data* when needed.
    For prayer time just display {'Fajr', 'Duhr', 'Asr', 'Maghrib', 'Isha'} with the timing HH:MM. 
    For weather use only {'weathed_data'} tool and just display the temperature and the humidity of the city. 
    Never add responses that you're not asked for.
    Format the results of {'prayer_times', 'weather_data'} to display the information in a clear and readable structure such as:
        - Infos1
        - Infos2
        ...
    For other responses make a clear text providing the necessary informations.
    """)

def prayer_times(city: str) -> dict:
    
    # Using Aladhan API to fetch actual prayer times
    url = f"https://api.aladhan.com/v1/timingsByAddress?address={city}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data']['timings']
    except Exception as e:
        return {"Error": str(e)}

def weather_data(city: str) -> dict:
    
    # Using Weather API to fetch actual weather data
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"      
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['main']
    except Exception as e:
        return {"Error": str(e)}

available_functions = {
    "prayer_times": prayer_times,
    "weather_data": weather_data,
}

def start_chat():
    print("--- Groq Chat Initialized ---")

    # Initialize conversation history 
    memory_history = [
        {
            "role": "system",
            "content": instraction()
        }
    ]

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Chat ended.")
            break

         # Describe the tool to the LLM (The "Manual")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "prayer_times",
                    "description": "Get the prayer times for a specific city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city",
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_data",
                    "description": "Get the weather times for a specific city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city",
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
        ]
        try:
            # Add the user's message to the history
            memory_history.append({"role": "user", "content": user_input})

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=memory_history,
                temperature=0.7,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = completion.choices[0].message
            # Append the assistant's message (which might contain tool calls) to history
            memory_history.append(response_message)
            
            tool_calls = getattr(response_message, 'tool_calls', None)  # Fetches an attribute from an object by name as a string.
            
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name in available_functions:
                        print(f"\n[*] Calling tool {function_name} with arguments: {function_args}")
                        function_response = available_functions[function_name](city=function_args.get("city"))
                        memory_history.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(function_response),
                            }
                        )
                
                # Get the final response after applying tool results
                second_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=memory_history,
                )
                print(f"\n{second_response.choices[0].message.content}\n")
            else:
                print(f"\n{response_message.content}\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    start_chat()
