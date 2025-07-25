from typing import Annotated, Dict, Iterable, Sequence, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load the env
load_dotenv()


# Create the langgraph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Create the tools
@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds two numbers together"""

    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """This is an subtraction function that subtract two numbers together"""

    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """This is an multiplication function that multiplies two numbers together"""

    return a * b


# Creating the list of tools
tools = [add, subtract, multiply]

# Initialize the llm and bind tools to it
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


# Create the langgraph nodes
def model_call(state: AgentState) -> AgentState:
    """This function will call the LLM"""

    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability"
    )
    messages = [system_prompt] + list(state["messages"])
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """This function will decide what to do next"""

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"


# Declare the StateGraph from langgraph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("our_agent", model_call)

# Add tool node to the graph
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

# Add edges to the graph
graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "our_agent")

# Compile the graph
app = graph.compile()


# Helper function
def print_stream(stream: Iterable[Dict[str, Sequence[Union[BaseMessage, tuple]]]]):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        elif isinstance(message, BaseMessage):
            message.pretty_print()
        else:
            print(message)


# Call the agent
initial_state = AgentState(messages=[HumanMessage(content="Add 40 + 12 and then multiply the result by 6")])
print_stream(app.stream(initial_state, stream_mode="values"))
