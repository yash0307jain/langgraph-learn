from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Load the env
load_dotenv()


# Create the langgraph state
class AgentState(TypedDict):
    messages: List[HumanMessage]


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")


# Create the langgraph nodes
def process(state: AgentState) -> AgentState:
    """This function calls LLM and prints it"""

    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}\n")
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

# Create the chatbot loop
user_input = input("Enter: ")
while user_input != "exit":
    # Invoke the agent
    intial_state = AgentState(messages=[HumanMessage(content=user_input)])
    result = agent.invoke(intial_state)

    # Take the user input again
    user_input = input("Enter: ")
