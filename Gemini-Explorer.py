import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part, Content, ChatSession

# Initialize Vertex AI project
project = "ethereal-runner-425400-u1"
vertexai.init(project=project)

# Configuration for the generative model
config = generative_models.GenerationConfig(
    temperature=0.4
)

# Create the generative model
model = GenerativeModel(
    "gemini-pro",
    generation_config=config
)
chat = model.start_chat()

# Streamlit app title
st.title("Gemini Explorer")

# Function to handle chat with LLM
def llm_function(chat: ChatSession, query):
    # Check message alternation
    if not validate_message_sequence(st.session_state.messages):
        st.error("Message sequence error: Messages must alternate between user and model, and the last message must be from the model.")
        return
    
    response = chat.send_message(query)
    output = response.candidates[0].content.parts[0].text

    with st.chat_message("model"):
        st.markdown(output)

    # Append the user's query
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Append the model's response
    st.session_state.messages.append(
        {
            "role": "model",
            "content": output
        }
    )

# Function to validate the message sequence
def validate_message_sequence(messages):
    # Ensure messages alternate and the last message is from the model
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i-1]["role"]:
            return False
    if len(messages) > 0 and messages[-1]["role"] == "user":
        return False
    return True

# Initialize session state messages if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initial message handling
if len(st.session_state.messages) == 0:
   
    initial_prompt = "Introduce yourself as a explorer assistant, ReX, powered by Google Gemini."
    llm_function (chat, initial_prompt)

# Display chat history
for index, message in enumerate(st.session_state.messages):
    content = Content(
        role=message["role"],
        parts=[Part.from_text(message["content"])]
    )
    if index != 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat.history.append(content)

# User input handling
query = st.chat_input("Gemini Explorer")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(chat, query)
