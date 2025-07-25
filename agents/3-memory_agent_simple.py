from typing import List, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Load the env
load_dotenv()


# Create the langgraph state
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")


# Create the langgraph nodes
def process(state: AgentState) -> AgentState:
    """This function calls LLM and prints it"""

    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}\n")

    state["messages"].append(AIMessage(content=response.content))
    return state


# Declare the StateGraph from langgraph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("process", process)

# Add egdes to the graph
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compile the graph
agent = graph.compile()

# Read the history data from the text file
conversation_history: List = []
try:
    with open("data/chatbot_history.txt", "r", encoding="utf-8") as file:
        data = file.read()
        conversation_history = [data]
except Exception:
    print("There is not history available now\n")

# Create the chatbot loop
user_input = input("Enter: ")
while user_input != "exit":
    # Store the user input as Human message
    conversation_history.append(HumanMessage(content=user_input))

    # Invoke the agent
    intial_state = AgentState(messages=conversation_history)
    result = agent.invoke(intial_state)

    # Update the conversation history by taking the result, which is the state in the graph
    conversation_history = result["messages"]

    # Take the user input again
    user_input = input("Enter: ")

# Store the history in a text file
with open("data/chatbot_history.txt", "w", encoding="utf-8") as file:
    file.write("Your conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        if isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation")

print("Conversation saced to chatbot_history.txt")
