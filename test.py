from langchain_core.messages import HumanMessage
from analyze_agent.agent import analyze_agent, run_agent
from analyze_agent.tools import encode_image

image = "image.jpg"
base64_img = encode_image(image)

# Build a proper multimodal message
message = HumanMessage(content=[
    {"type": "text", "text": "recreate the kolam image.image path is image.jpg and output path is output_{count}.jpg and count from 1 to the number of generation"}
])

# Directly invoke the agent (bypassing the simple run_agent wrapper)
response = run_agent(message)
# print(response)
breakpoint()
