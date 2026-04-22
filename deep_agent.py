from langchain_groq import ChatGroq 
from dotenv import load_dotenv  
#dotenv is used to load environment variables from .env file into your program
import json

#LOAD MODEL

print("Loading model.. This may take few mins..")

load_dotenv() #loading dotenv
llm = ChatGroq(
    model = "llama-3.1-8b-instant"
)
print("Model loaded successfully.\n")

#PLANNING TOOL

def create_plan(goal): # Gets input from user
    prompt = f"""
You are a professional curriculum designer.
Create exactly 5 clear, practical learning steps for:

GOAL: {goal}

Rules:
- Create exactly 5 steps.
- Each step must begin with "Step X:"
- Each step must be specific and actionable.
- Each step should build logically on the previous one.
- Each step must be on a new line
- Do NOT repeat the goal.
- Keep steps short but meaningful.

Generate the 5 steps in below format:

Step 1:
Step 2:
Step 3:
Step 4:
Step 5:

"""
    response = llm.invoke(prompt) # output is saved here
    return response.content.strip() # strip removes unwanted blankspaces

# SUB AGENT

def writing_agent(task):
    prompt = f"""
You are a professional technical writer.

Your job:
- Turn the task into well-structured educational content within 50 words
- Write clearly for beginners
- Use headings and short paragraphs
- Be practical and actionable

TASK:
{task}
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# MEMORY SYSTEM

def save_memory(filename, data):
    with open(filename, "w") as f: # open file
        json.dump(data, f, indent=2) # dump the file in json format

# DEEP AGENT ORCHESTRATOR

def deep_agent(goal):
    print("Goal: ", goal)

    #Step 1: Create plan
    plan = create_plan(goal)
    print("\nGenerated Plan:\n", plan)

    steps = plan.split("\n")
    results = []

    #Step 2: Execute steps
    print("\nSub Agent output\n")
    for step in steps:  #iterates 5 steps
        step = step.strip()
        if not step:
            continue

        result = writing_agent(step)
        print("Completed:", result)
        results.append(result) #results is stored in results array

    #Step 3 : Save memory
    save_memory("memory.json", results)

    print("\nAll tasks completed. Memory saved to memory.json")

# RUN THE AGENT

if __name__ == "__main__":   #Input
    user_goal = "Create a beginner guide to learn Music"
    deep_agent(user_goal)
