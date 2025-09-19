from typing import Sequence, TypedDict, List, Union, Annotated
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from datetime import datetime

import asyncio
from kuksa_client.grpc.aio import VSSClient
from kuksa_client.grpc import Datapoint

import os
import time

import json

load_dotenv()

import sqlite3

try:
    connection = sqlite3.connect("history.db")
    cursor = connection.cursor()
    print("✅ Your database is ready. ")
except:
    print("❌ Something is wrong, can't open database. ")

try:
    get_date = datetime.now().strftime("%Y_%m_%d")
    table_name = f"'{get_date}_table'"
    check_table = cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name={table_name}")
    if check_table.fetchone():
        print("✅ Table exit")
    else:
        cursor.execute(f"""
                            CREATE TABLE {table_name} (
                                human_messages str,
                                ai_messages int
                            )
                        """)
        connection.commit()
        print("✅ Successfull created table")
except Exception as e:
    print("Facing error when try to creating or connecting to table: ", e)

try:
    with open('define.json') as F:
        configure = json.load(F)
except Exception as e:
    print("Error: " , e)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

vss = VSSClient(configure["ip_address"], configure["port"])
    

async def tell(type):
    try:
        async with vss as client:
            thing = await client.get_target_values([
                f"{type}"
            ])
            state = thing[f"{type}"].value
            
        return state
    except:
        return False

async def set(new_state: Union[bool, int], type: str):
    try:
        async with vss as client:
            success = await client.set_target_values({
                f"{type}": Datapoint(new_state)
            })
            return success
    except:
        return False

@tool
def teller(api: str):
    """ 
        Tool to check the state or current value of thing based on the api that user want.
        Choose the api based on the user demand
        Args:
            api (str): The api based on what user say.
    """

    state = asyncio.run(tell(api))
    return state

@tool
def setter(state: Union[bool, int], api: str):
    """
        Tool to set the vehicle api.
        pass the api that the user demand you to change in for api
        pass the value that user demand you to change to in state
        Args:
            lights_type : The api based on what user say.
            state : The next state that user want the api to be.
        Returns:
            bool: True if lights was set succesffully, False otherwise
    """
    result = asyncio.run(set(state, api))
    return result


@tool
def time_teller():
    """Returns the current time."""
    return datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

tools = [time_teller, teller, setter]

from main import ask_and_answer
from langchain_core.prompts import ChatPromptTemplate
from compineVector import retriever

model = ChatOllama(model="llama3.2:3b").bind_tools(tools)

template = """
    You are an exeprt in answering questions about a vehicle, if user doesn't ask about any involve vehicle api then answer normally
    if they ask for api then prioritize giving the whole api and you should you "." in api not "/", however answer as straight to the point as posible
    If they want to change the state of the vehicle or status of vehicle attachment tool then answer arcoding to the example.
    Here are some knowledge you need to know about yoursefl: {name}, {definition}

    Here are some relevant infor: {information}

    Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model



def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
        There are couple of tool that you can you to help user archive their goal. Only call tool when you feel it will help user archiving their goal, don't call tool irresponsibly, and only call 1 tool per time
        The time_teller is the most simplest it help you to get current time.
        If user want to change the state of api or ECUs in the call, then you will call setter, then pass api and value arcoding to what user say to that then execute it.
        If user simply want to know the current state or current value of an api or ECUs in vehicle, then call teller, then pass api in it which will help you recall the value of that api
    """)

    last_user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_input = msg.content.lower()
            break

    allowed_keywords = ["time", "clock", "date", "today", "set", "change"]

    is_tool_needed = any(keyword in last_user_input for keyword in allowed_keywords)

    if is_tool_needed:
        last_message = state['messages'][-1].content
        information = retriever.invoke(last_message)
        
        # Assign result to response
        response = chain.invoke({
            "information": information,
            "question": last_message,
            "name": configure["name"],
            "definition": configure["definition"],
            "system_prompt": system_prompt,
        })

        print("Thinking...")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print("\nAI is making a tool call")
            for call in response.tool_calls:
                print(f"→ Tool: {call['name']}, Arguments: {call['args']}")
        else:
            response = ask_and_answer(state["messages"][-1].content, configure["name"], configure["definition"])
            print("AI: ", response)

        state["messages"].append(response)
        return state

    else:
        result = ask_and_answer(state["messages"][-1].content, configure["name"])
        print("AI: ", result)
        
        state["messages"].append(AIMessage(content= result))

        return state


#def model_call(state: str) -> str:
#    system_prompt = SystemMessage(content="""
#        There are couple of tool that you can you to help user archive their goal. Only call tool when you feel it will help user archiving their goal, don't call tool irresponsibly, and only call 1 tool per time
#        The time_teller is the most simplest it help you to get current time.
#        If user want to change the state of api or ECUs in the call, then you will call setter, then pass api and value arcoding to what user say to that then execute it.
#        If user simply want to know the current state or current value of an api or ECUs in vehicle, then call teller, then pass api in it which will help you recall the value of that api
#    """)

#    last_user_input = ""
#    for msg in reversed(state["messages"]):
#        if isinstance(msg, HumanMessage):
#            last_user_input = msg.content.lower()
#            break
    
#    last_message = state['messages'][-1].content
#    information = retriever.invoke(last_message)

#    response = chain.invoke({
#            "information": information,
#            "question": last_message,
#            "name": configure["name"],
#            "definition": configure["definition"],
#            "system_prompt": system_prompt,
#        })

#    print("Thinking...")

#    if hasattr(response, "tool_calls") and response.tool_calls:
#        print("\nAI is making a tool call")
#        for call in response.tool_calls:
#            print(f"→ Tool: {call['name']}, Arguments: {call['args']}")
#    else:
#        response = ask_and_answer(last_message, configure["name"], configure["definition"])
#        print("AI: ", response)

#    state["messages"].append(response)
#    return state


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"

    
graph = StateGraph(AgentState)
graph.add_edge("tools", "our_agent")
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_edge(START, "our_agent")
graph.add_edge("our_agent", END)

agent = graph.compile()

history = []

agent.invoke({"messages": [HumanMessage(content= "Hello")]})

while True:
    user_input = input("\nEnter: ")

    if user_input in ["end", "exit", "clode", "goodbye"]:
        print("Turning model off...")
        break
    history.append(HumanMessage(content= user_input))
    result = agent.invoke({"messages" : history})
    try:
        if len(history) > 0:
            # Find the last AI message
            ai_mess = ""
            for msg in reversed(history):
                if isinstance(msg, AIMessage):
                    ai_mess = msg.content
                    break

            # Store user input and AI response
            cursor.execute(f"INSERT INTO {table_name} VALUES (?,?)", (user_input, ai_mess))
            connection.commit()

    except Exception as e:
        print("Failed to store conversation: ", e)
    history = result["messages"]
connection.close()
print(f"\n✅ Successfull turn llm model off")