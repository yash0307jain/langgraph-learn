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


# Initialize the llm
llm = ChatOpenAI(model="gpt-4o-mini")


# Create the langgraph nodes
def user_input(state: AgentState) -> AgentState:
    """This function will take the user input and pass to the next node"""

    user_input = input("Enter: ")
    state["messages"].append(HumanMessage(content=user_input))
    return state


def should_continue(state: AgentState) -> str:
    """This function decdes what to do next"""

    user_input = state["messages"][-1].content
    if user_input == "exit":
        return "exit"
    return "continue"


def process(state: AgentState) -> AgentState:
    """This function calls the LLM and stores the Human and AI message"""

    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}\n")

    state["messages"].append(AIMessage(content=response.content))
    return state


# Declare the StateGraph from langgraph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("user_input", user_input)
graph.add_node("process", process)


# Add edges to the graph
graph.add_edge(START, "user_input")
graph.add_conditional_edges(
    "user_input",
    should_continue,
    {
        "continue": "process",
        "exit": END,
    },
)
graph.add_edge("process", "user_input")

# Compile the graph
agent = graph.compile()

# Invoke the agent
initial_state = AgentState(messages=[])
agent.invoke(initial_state)
