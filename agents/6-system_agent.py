import json
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Load the environment variables
load_dotenv()


# Create the user details model
class UserDetails(BaseModel):
    user_name: str = Field(description="Name of the user")
    goal: str = Field(description="User goal that he/she want to achieve using the system")
    qualities_score: Dict[str, float] = Field(description="User qualities along with rating from 0 to 5")
    weakness: List[str] = Field(description="User weaknesses that he/she want to improve to grow")
    time_available: str = Field(description="Time available in a day")


class SystemTask(BaseModel):
    name: str = Field(description="Name of the task")
    time: str = Field(description="For how long this task has to be perform in a day, in hours or minutes")
    xp: str = Field(description="How much xp points it will earn after completing this task")


class SystemTasks(BaseModel):
    tasks: List[SystemTask] = Field(
        description="This is going to be the list of SystemTask's, "
        "provide the tasks to be done in proper sequence as well"
    )


# Create the Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_details: Optional[UserDetails]
    tasks: Optional[SystemTasks]


# Create tools
# @tool
# def read_the_user_details() -> UserDetails:
#     """
#     This tool is going to read the user data from the user_details file and
#     convert into a UserDetails pydantic class and return it
#     """
#     data = None
#     with open("data/user_details.json", "r") as file:
#         data = file.read()
#     json_data = json.loads(data)

#     user_details = UserDetails(**json_data)
#     return user_details


@tool
def save_the_system_task(tasks: SystemTasks) -> None:
    """This tools is going to take the SystemTasks and going to save into the file"""
    print(tasks)
    with open("data/system_tasks.json", "w") as file:
        tasks_json = tasks.model_dump()
        json.dump(tasks_json, file)


tools = [save_the_system_task]


# Create the llm instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


# Create the nodes of the graph
def input_user_details(state: AgentState) -> AgentState:
    """This node will going to ask for the user details"""

    # user_name = input("Please provide your name: ")
    # goal = input("What is your goal: ")

    # qualities_score = {}
    # print("What skills and qualities you have along with score between 0 to 5 (enter 'quit' to exit)")
    # while True:
    #     quality = input("Quality: ")
    #     if quality == "quit":
    #         break
    #     score = input("Quality score: ")
    #     qualities_score[quality] = float(score)

    # weakness = []
    # print("What is your weakness that you want to improve using the system (enter 'quit' to exit)")
    # while True:
    #     _weakness = input("Weakness: ")
    #     if _weakness == "quit":
    #         break
    #     weakness.append(_weakness)

    # time_available = input("Time available in a day: ")

    # user_details = UserDetails(
    #     user_name=user_name,
    #     goal=goal,
    #     qualities_score=qualities_score,
    #     weakness=weakness,
    #     time_available=time_available,
    # )

    data = None
    with open("data/user_details.json", "r") as file:
        data = file.read()
    json_data = json.loads(data)

    user_details = UserDetails(**json_data)
    return {"messages": [], "user_details": user_details}  # type: ignore


def create_user_system_tasks(state: AgentState) -> AgentState:
    """This node will going to take the user_details state and using this going to create user system tasks."""

    user_details: UserDetails = state["user_details"]  # type: ignore
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a system like there in one in Solo Leveling, and you have to create the system
                for user based on user details""",
            ),
            ("human", "Create a detailed plan, User Details: {user_details}"),
        ]
    )

    chain = prompt | llm.with_structured_output(schema=SystemTasks)
    system_tasks = chain.invoke({"user_details": user_details})
    system_tasks = cast(SystemTasks, system_tasks)

    tool_call_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You need to save these tasks using the save_the_system_task tool."),
            ("human", "Save these tasks: {tasks}"),
        ]
    )

    chain = tool_call_prompt | llm.bind_tools(tools=tools)
    response = chain.invoke({"tasks": system_tasks})

    # save_the_system_task.invoke({"tasks": system_tasks})

    return {
        "messages": [response],
        "user_details": user_details,
        "tasks": system_tasks,
    }


# Declare the state graph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("input_user_details", input_user_details)
graph.add_node("create_user_system_tasks", create_user_system_tasks)
graph.add_node("tools", ToolNode(tools=tools))

# Add edges to the graph
graph.add_edge(START, "input_user_details")
graph.add_edge("input_user_details", "create_user_system_tasks")
graph.add_edge("create_user_system_tasks", "tools")
graph.add_edge("tools", END)

# Compile the graph
agent = graph.compile()

# Invoke the graph
initial_state = AgentState(messages=[])  # type: ignore
response = agent.invoke(initial_state)
