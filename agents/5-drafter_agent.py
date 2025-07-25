from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load the env
load_dotenv()


# This is a global variable to store document content
document_content: str = ""


# Create the langgraph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Create the tools
@tool
def update(content: str) -> str:
    """Update the document with the provided content"""

    global document_content
    document_content = content
    return f"Document has been updated sucessfully! The current content is\n{document_content}"


@tool
def save(file_name: str) -> str:
    """Save the current document in the text file and finish the process

    Args:
        file_name: Name of the text file
    """

    global document_content

    if not file_name.endswith(".txt"):
        file_name = f"{file_name}.txt"

    try:
        with open(f"data/{file_name}", "w") as file:
            file.write(document_content)
        return f"Document has been saved successfully to the '{file_name}'."
    except Exception as e:
        return f"Error saving the document: {str(e)}"


# Creating the list of tools
tools = [update, save]

# Initialize llm and bind the tools to it
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools=tools)


# Create the langgraph nodes
def our_agent(state: AgentState) -> AgentState:
    """This function will take the user input and invoke the model and return the response"""
    system_prompt = SystemMessage(
        content=f"""
            You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

            - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
            - If the user wants to save and finish, you need to use the 'save' tool.
            - Make sure to always show the current document state after modifications.

            The current document content is:{document_content}
        """
    )
    if not state["messages"]:
        user_input = "I'm ready to help you to update the document. What would you like to create?"
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")

    user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if isinstance(response, AIMessage) and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]
    if not messages:
        return "continue"

    # Ths looks for the most recent tool message
    for message in reversed(messages):
        # Check if this tool message resulting from the save tool
        if (
            (isinstance(message, ToolMessage))
            and "saved" in str(message.content).lower()
            and "document" in str(message.content).lower()
        ):
            return "end"

    return "continue"


def print_messages(messages):
    """The function to print in more readable format"""

    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# Declare the StateGraph from langgraph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

# Add edges to the graph
graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

# Compile the graph
agent = graph.compile()


# Run the drafter agent
def run_drafter_agent():
    print("\n ===== DRAFTER =====")

    state = AgentState(messages=[])

    for step in agent.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_drafter_agent()
