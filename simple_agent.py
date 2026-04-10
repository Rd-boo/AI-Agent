import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. IMPORT MEMORY SAVER
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_retries=3,  # It will automatically wait and try again up to 3 times
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 2. INITIALIZE MEMORY
memory = MemorySaver()

# 3. COMPILE WITH CHECKPOINTER
app = workflow.compile(checkpointer=memory)


def run_chat():
    print("--- Agent IA Active (Type 'quit' to stop) ---")

    # 4. DEFINE A THREAD ID
    # This acts as the session ID. LangGraph will automatically save
    # and load the state for this specific thread.
    config: RunnableConfig = {"configurable": {"thread_id": "tool_session"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # 5. PASS THE CONFIG TO THE STREAM
        events = app.stream(
            {"messages": [("user", user_input)]}, config=config, stream_mode="values"
        )

        for event in events:
            if "messages" in event:
                last_message = event["messages"][-1]
                # Only print if it's the AI's turn, otherwise it echoes the user input
                if hasattr(last_message, "content") and last_message.type == "ai":
                    print(f"Agent: {last_message.content}")


if __name__ == "__main__":
    run_chat()
