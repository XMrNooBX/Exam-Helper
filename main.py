from langchain_mistralai import ChatMistralAI
from st_multimodal_chatinput import multimodal_chatinput
from streamlit_carousel import carousel
import streamlit as st
from PIL import Image
import base64
import io
import re

import utils

# --- Streamlit Configuration and Styling ---
st.set_page_config(page_title="Bhala Manus", page_icon="🌟", layout="wide")

# --- Enhanced CSS for a Modern Look ---
st.markdown(
    """
<style>
/* --- Base Styles --- */
.main {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #454545;
    color: #fff;
}

/* --- Header Styling --- */
.header {
    text-align: center;
    color: #00ffff; /* Bright Cyan for a Techy Feel */
    font-size: 48px;
    font-weight: bold;
    text-shadow: 0 0 15px #00ffff;
    animation: glow 2s ease-in-out infinite alternate;
}

/* --- Subheader Styling --- */
.subheader {
    color: #ffcc00; /* Vibrant Yellow */
    font-size: 22px;
    text-align: center;
    margin-top: 10px;
}

/* --- Button Styling --- */
.stButton>button {
    background-color: #ff4b4b; /* Energetic Red */
    color: white;
    border-radius: 25px;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #cc0000;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
}

/* --- Text Input Styling --- */
.stTextInput>div>input {
    background-color: #fff;
    border-radius: 20px;
    border: 2px solid #ff4b4b;
    padding: 12px;
    font-size: 18px;
    color: #333;
}

/* --- Checkbox Styling --- */
.stCheckbox>div>label {
    font-size: 18px;
    color: #ff4b4b;
    font-weight: bold;
}

/* --- Chat Input Styling --- */
.stChatInput>div>input {
    background-color: #f5f5f5;
    border: 2px solid #ff4b4b;
    border-radius: 20px;
    padding: 12px;
    font-size: 18px;
}

/* --- Markdown Styling --- */
.stMarkdown {
    font-size: 18px;
    line-height: 1.6;
}

/* --- Chat Message Styling --- */
.stChatMessage > div {
    border-radius: 20px;
    padding: 16px;
    margin: 10px 0;
    font-family: 'Segoe UI', sans-serif;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.stChatMessage > div:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

/* --- User Message Styling --- */
.stChatMessage > div.user {
    background-color: rgba(13, 9, 10, 0.8); /* Dark with Opacity */
    color: #fff;
    border-left: 5px solid #ff4b4b;
    margin-left: 40px;
    margin-right: 80px;
    animation: slideInRight 0.5s ease-out;
}

/* --- Assistant Message Styling --- */
.stChatMessage > div.assistant {
    background-color: rgba(70, 40, 90, 0.8); /* Royal Purple with Opacity */
    color: #fff;
    border-left: 5px solid #00ffff;
    margin-left: 80px;
    margin-right: 40px;
    animation: slideInLeft 0.5s ease-out;
}

/* --- Sidebar Styling --- */
.sidebar .sidebar-content {
    background-color: #333;
    padding: 20px;
    border-radius: 10px;
}

.sidebar .stTextInput>div>input {
    background-color: #fff;
    border-radius: 15px;
    border: 2px solid #ff4b4b;
    padding: 10px;
    font-size: 16px;
    color: #333;
}

/* --- Animations --- */
@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes slideInLeft {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #00ffff;
    }
    to {
        text-shadow: 0 0 20px #00ffff, 0 0 30px #00ffff;
    }
}

/* --- App Background --- */
.stApp {
    background: linear-gradient(135deg, #141e30, #243b55); /* Dark Gradient */
    animation: gradientAnimation 15s ease infinite;
    background-size: 200% 200%;
    background-attachment: fixed;
}

@keyframes gradientAnimation {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* --- Scrollbar Styling --- */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #ff4b4b;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #cc0000;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Header and Subheader ---
st.markdown('<div class="header">🌟 No Back Abhiyan 🌟</div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Padh le yaar...</p>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<h3 style="color: #00ffff;">Configuration</h3>', unsafe_allow_html=True)
    index_name = st.selectbox("Doc Name", options=["ai-docs"], index=0, help="Select the name of the Documents to use.")
    groq_api_key = st.text_input("LLM API Key", type="password", help="Enter your groq API key.")
    model = st.selectbox(
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
        st.error("Please enter the LLM API key to proceed!")

    use_web = st.checkbox("Allow Internet Access", value=True)
    use_vector_store = st.checkbox("Use Documents", value=True)
    use_chat_history = st.checkbox("Use Chat History (Last 2 Chats)", value=False)

    if use_chat_history:
        use_vector_store, use_web = False, False

    # --- Instructions ---
    st.markdown(
        """
    ---
    **Instructions:**  
    Get your *Free-API-Key*  
    From **[Groq](https://console.groq.com/keys)**

    --- 
    Kheliye *meating - meeting*
    """
    )

# --- API Keys ---
api_keys = {
    "pinecone": "pcsk_7T5yD_5dUhax1xCeRv6Tm1MdiDpBkGTpv41tcUzsH2VG671YW2gFaQYcGqF57QY3BFZWn",
    "google": "AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw",
    "groq": groq_api_key,
}
subject = index_name.split('-')[0]
# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state["last_query"] = "El Gamal"

# --- Initialize Vector Store and Language Model ---
if "vector_store" not in st.session_state and groq_api_key:
    vector_store = utils.get_vector_store(index_name, api_keys)
    st.session_state["vector_store"] = vector_store
    st.session_state["index_name"] = index_name
    st.success(f"Successfully connected to the Vector Database: {index_name}! Let's go...")
else:
    vector_store = st.session_state.get("vector_store")

if "index_name" in st.session_state and st.session_state["index_name"] != index_name:
    vector_store = utils.get_vector_store(index_name, api_keys)
    st.session_state["vector_store"] = vector_store
    st.session_state["index_name"] = index_name
    st.success(f"Successfully connected to the Vector Database: {index_name}! Let's go...")

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

# --- Fallback Model ---
llmx = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3,
    api_key="fCf6gpwY6D3Z2mJlHAHxcg4xhp6f6Xcz",
)

# --- Function to Display Chat History ---
def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👼"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"], avatar="🧑‍🏫"):
                st.write(message["content"])

# --- Display Chat History ---
display_chat_history()

# --- Main Chat Interaction Loop ---
if groq_api_key:
    with st.container():
        user_inp = multimodal_chatinput()

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
                subject,
            )
            with st.spinner(":green[Combining jhol jhal...]"):
                assistant_response = utils.respond_to_user(
                    question + user_inp["text"], context, llm, subject
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )

            with st.chat_message("user", avatar="👼"):
                st.write(question + user_inp["text"])
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.write(assistant_response)
