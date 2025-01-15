from langchain_mistralai import ChatMistralAI
from st_multimodal_chatinput import multimodal_chatinput
from streamlit_carousel import carousel
import streamlit as st
from PIL import Image
import base64
import io
import re

import utils

# Streamlit Configuration and Styling
st.set_page_config(page_title="Bhala Manus", page_icon="🌟")

css = """
<style>
/* Make header background transparent */
.stApp > header {
    background-color: transparent !important;
}

/* Apply animated gradient background */
.stApp {
    background: linear-gradient(45deg, #3a5683 10%, #0E1117 45%, #0E1117 55%, #3a5683 90%);
    animation: gradientAnimation 20s ease infinite;
    background-size: 200% 200%;
    background-attachment: fixed;
}

/* Keyframes for smooth animation */
@keyframes gradientAnimation {
    0% {
        background-position: 0% 0%;
    }
    50% {
        background-position: 100% 100%;
    }
    100% {
        background-position: 0% 0%;
    }
}
</style>

"""
st.markdown(css, unsafe_allow_html=True)

# HTML content
html = """
<section id="up"></section>
<section id="down"></section>
<section id="left"></section>
<section id="right"></section>
"""

def render_frontend():
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

st.markdown(
    """
<style>
.main {
    font-family: 'Arial', sans-serif;
    background-color: #454545;
    color: #fff;
}
.header {
    text-align: center;
    color: #47fffc;
    font-size: 36px;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    font-size: 16px;
}
.stTextInput>div>input {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #4CAF50;
    padding: 10px;
    font-size: 16px;
}
.stCheckbox>div>label {
    font-size: 16px;
    color: #4CAF50;
}
.stChatInput>div>input {
    background-color: #e8f5e9;
    border: 1px solid #81c784;
}
.stMarkdown {
    font-size: 16px;
}

.stChatMessage > div {
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0; /* General spacing for top and bottom */
    font-family: "Arial", sans-serif;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15); /* Subtle shadow for depth */
}

/* User messages (Dark with 60% opacity, right indent) */
.stChatMessage > div.user {
    background-color: rgba(13, 9, 10, 0.6); /* Smoky Black with 60% opacity */
    color: #EAF2EF; /* Mint Cream for text contrast */
    border-left: 4px solid #361F27; /* Dark Purple accent */
    margin-left: 32px; /* General left margin */
    margin-right: 64px; /* Right indent for user messages */
}

/* Assistant messages (Light Lavender with 40% opacity, left indent) */
.stChatMessage > div.assistant {
    background-color: rgba(70, 40, 90, 0.6); /* Royal purple for assistant messages */
    color: #F5D7FF; /* Lavender text for a softer, friendly vibe */
    border-left: 4px solid #BB8FCE;
    margin-left: 64px; /* Left indent for assistant messages */
    margin-right: 32px; /* General right margin */
}

/* Hover effects for smooth interaction */
.stChatMessage > div:hover {
    transform: scale(1.005);
    transition: transform 0.2s ease-in-out;
}

</style>
""",
    unsafe_allow_html=True,
)

# Header and message below it
st.markdown(
    '<div class="header">🌟No Back Abhiyan </div>', unsafe_allow_html=True
)
st.markdown(
    '<p style="color: #dcfa2f; font-size: 18px; text-align: center;">Padh le yaar...</p>',
    unsafe_allow_html=True,
)

# Sidebar and configuration
st.sidebar.markdown(
    """<h3 style="color: cyan;">Configuration</h3>""", unsafe_allow_html=True
)
index_name = st.sidebar.selectbox(
    "Doc Name", options=["os-docs"], index=0, help="Select the name of the Documents to use."
)
groq_api_key = st.sidebar.text_input(
    "LLM API Key", type="password", help="Enter your groq API key."
)
model = st.sidebar.selectbox(
    "Select Model",
    options=[
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-90b-vision-preview",
    ],
    index=0,
    help="Select the model to use for LLM inference.",
)
if not groq_api_key:
    st.sidebar.markdown(
        "<p style='color: #f44336;'>Please enter the LLM API key to proceed!</p>",
        unsafe_allow_html=True,
    )
    st.warning("Please enter the LLM API key to proceed!")

use_web = st.sidebar.checkbox("Allow Internet Access", value=True)
use_vector_store = st.sidebar.checkbox("Use Documents", value=True)
use_chat_history = st.sidebar.checkbox(
    "Use Chat History (Last 2 Chats)", value=False
)

if use_chat_history:
    use_vector_store, use_web = False, False

# Instructions
st.sidebar.markdown(
    """
---
**Instructions:**  
Get your *Free-API-Key*  
From **[Groq](https://console.groq.com/keys)**

--- 
Kheliye *meating - meeting*
"""
)

# API keys for various services
api_keys = {
    "pinecone": "pcsk_7T5yD_5dUhax1xCeRv6Tm1MdiDpBkGTpv41tcUzsH2VG671YW2gFaQYcGqF57QY3BFZWn",
    "google": "AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw",
    "groq": groq_api_key,
}

# Initialize vector store and language model if not already in session state
if "vector_store" not in st.session_state and groq_api_key:
    vector_store = utils.get_vector_store(index_name, api_keys)
    st.session_state["vector_store"] = vector_store
    st.session_state["index_name"] = index_name
    st.success(
        f"Successfully connected to the Vector Database: {index_name}! Let's go..."
    )
else:
    vector_store = st.session_state.get("vector_store")


if "index_name" in st.session_state and st.session_state["index_name"] != index_name:
    vector_store = utils.get_vector_store(index_name, api_keys)
    st.session_state["vector_store"] = vector_store
    st.session_state["index_name"] = index_name
    st.success(
        f"Successfully connected to the Vector Database: {index_name}! Let's go..."
    )

if groq_api_key:
    if "llm" not in st.session_state:
        llm = utils.get_llm(model, api_keys)
        st.session_state["llm"] = llm
        st.session_state["model"] = model
        st.session_state["api_key"] = groq_api_key
    else:
        llm = st.session_state["llm"]

if "api_key" in st.session_state and "model" in st.session_state:
    if groq_api_key != st.session_state["api_key"] or model != st.session_state["model"]:
        llm = utils.get_llm(model, api_keys)
        st.session_state["llm"] = llm
        st.session_state["model"] = model
        st.session_state["api_key"] = groq_api_key

# Fallback model
llmx = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3,
    api_key="RScM7WQKY4RtCVOOj49MWYqRVQB3zl9Y",
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state["last_query"] = "El Gamal"

# Function to display chat history
def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👼"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"], avatar="🧑‍🏫"):
                st.write(message["content"])

display_chat_history()

# Main chat interaction loop
if groq_api_key:
    with st.container():
        user_inp = multimodal_chatinput()
    with st.container():
        if user_inp:
            if user_inp == st.session_state["last_query"]:
                st.stop()
            else:
                st.session_state["last_query"] = user_inp
            video_id = ""
            question = ""
            if user_inp["images"]:
                b64_image = user_inp["images"][0].split(",")[-1]
                image = Image.open(io.BytesIO(base64.b64decode(b64_image)))
                try:
                    question = utils.img_to_ques(image, user_inp["text"])
                except:
                    question = utils.img_to_ques(image, user_inp["text"], "gemini-2.0-flash-exp")
                soln = re.findall(
                    r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))?(?:https://youtu.be/([^\?\n]+))?",
                    user_inp["text"],
                )
                for match in soln:
                    video_id = match[0] or match[1]  # Use the first non-empty part
                    if video_id:  # Stop at the first valid match
                        break
                    else:
                        video_id = ""
                user_inp["text"] = ""
            if not video_id:
                soln = re.findall(
                    r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))?(?:https://youtu.be/([^\?\n]+))?",
                    user_inp["text"],
                )
                for match in soln:
                    video_id = match[0] or match[1]  # Use the first non-empty part
                    if video_id:  # Stop at the first valid match
                        break
                    else:
                        video_id = ""
            st.session_state.messages.append(
                {"role": "user", "content": question + user_inp["text"]}
            )
            with st.spinner(":green[Checking Requirements For Image]"):
                diagram_required=utils.check_for_diagram(question + user_inp["text"],llmx)
            if diagram_required.requires_diagram:
                    with st.spinner(":green[Generating Diagram]"):
                        try:
                            images = utils.search_images(diagram_required.search_query, 5)
                        except Exception as e:
                            st.warning(f"Unable to Generate Diagram Due to Error: {e}")
                            images=""
                    if images:
                       carousel(images, fade=True, wrap=True, interval=999000)
            else:
                st.info("No Diagram Required For This Query")
            with st.spinner(":green[Processing Youtube Video]"):
                if video_id:
                    st.success(
                        f"!! Youtube Link Found:- {video_id} , Summarizing Video"
                    )
                    try:
                        yt_response = utils.process_youtube(
                            video_id, question + user_inp["text"], llmx
                        )
                    except Exception as e:
                        yt_response = f"Unable to Process , Youtube Video Due to Transcript not available Error: {e}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": yt_response}
                    )
                    with st.chat_message("user", avatar="👼"):
                        st.write(question + user_inp["text"])
                    with st.chat_message("assistant", avatar="🧑‍🏫"):
                        st.write(yt_response)            
            if not video_id:
                context = utils.get_context(
                    question + user_inp["text"],
                    use_vector_store,
                    vector_store,
                    use_web,
                    use_chat_history,
                    llm,
                    llmx,
                    st.session_state.messages,
                )
                with st.spinner(":green[Combining jhol jhal...]"):
                    assistant_response = utils.respond_to_user(
                        question + user_inp["text"], context, llm
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )

                with st.chat_message("user", avatar="👼"):
                    st.write(question + user_inp["text"])
                with st.chat_message("assistant", avatar="🧑‍🏫"):
                    st.write(assistant_response)
