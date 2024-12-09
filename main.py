from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from PIL import Image
from st_multimodal_chatinput import multimodal_chatinput
import base64,io

import streamlit as st

st.set_page_config(page_title="Bhala Manus", page_icon="🌟")

st.markdown("""
    <style>
    .main {
        font-family: 'Arial', sans-serif;
        background-color: #f0f8ff;
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
    </style>
""", unsafe_allow_html=True)

# Header and message below it
st.markdown('<div class="header">🌟No Back Abhiyan </div>', unsafe_allow_html=True)
st.markdown('<p style="color: #dcfa2f; font-size: 18px; text-align: center;">Padh le yaar...</p>', unsafe_allow_html=True)

# Sidebar and configuration
st.sidebar.markdown("""<h3 style="color: cyan;">Configuration</h3>""", unsafe_allow_html=True)
index_name = st.sidebar.text_input("Doc Name", value="ml-docs", help="Enter the name of the index to use.")
groq_api_key = st.sidebar.text_input("LLM API Key", type="password", help="Enter your groq API key.")

if not groq_api_key:
    st.sidebar.markdown("<p style='color: #f44336;'>Please enter the LLM API key to proceed!</p>", unsafe_allow_html=True)



use_web = st.sidebar.checkbox("Allow Internet Access", value=True)
use_vector_store = st.sidebar.checkbox("Use Documents", value=True)
use_chat_history = st.sidebar.checkbox("Use Chat History (Last 2 Chats)", value=False)

if use_chat_history:
    use_vector_store, use_web = False, False


def img_to_ques(img,query):
    genai.configure(api_key="AIzaSyBGMk5yhUdGv-Ph5P6Y5rq7F3G56GQJbaw")
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    prompt = f"""Analyze the provided image and the query: "{query}". Based on the content of the image:
1. Extract or generate a question that aligns with the given query.
2. Identify and provide any relevant information or insights from the image that could help address or solve the problem described in the query.

Format your response as follows:

Question:  
[Generated question based on the image and query]  

Relevant Information:  
**optional, only if there are any**
[Key details or insights from the image that are relevant to solving the problem]
"""
    return model.generate_content([prompt, img]).text

# Instructions
st.sidebar.markdown("""
---
**Instructions:**  
Get your *Free-API-Key*  
From **[Groq](https://console.groq.com/keys)**

--- 
Kheliye *meating - meeting*
""")

# API Key Connection to Vector Database with feedback
if "vector_store" not in st.session_state and groq_api_key:
    pc = Pinecone(api_key="pcsk_6KAu86_9Zzepx9S1VcDmLRmBSUUNpPf4JRbE4BaoVmk36yW9R4nkjutPiZ3AjZvcyL4MVx")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw")
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    st.session_state["vector_store"] = vector_store
    st.success("Successfully connected to the Vector Database! let's go...")
else:
    vector_store = st.session_state.get("vector_store", None)

# LLM API Key check
if groq_api_key:
    if "llm" not in st.session_state:
        llm = ChatGroq(temperature=0.2, model="llama-3.1-70b-versatile", api_key=groq_api_key)
        st.session_state["llm"] = llm
    else:
        llm = st.session_state["llm"]

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Function to Clean RAG Data
def clean_rag_data(query, context, llm):
    system = """
        You are a Highly capable Proffesor of understanding the value and context of both user queries and given data. 
        Your Task for Documents Data is to analyze the list of document's content and properties and find the most important information regarding user's query.
        Your Task for ChatHistory Data is to analyze the given ChatHistory and then provide a ChatHistory relevant to user's query.
        Your Task for Web Data is to analyze the web scraped data then summarize only useful data regarding user's query.
        You Must adhere to User's query before answering.
        
        Output:
            For Document Data
                Conclusion:
                    ...
            For ChatHistory Data
                    User: ...
                    ...
                    Assistant: ...
            For Web Data
                Web Scarped Data:
                ...
    """

    user = """{context}
            User's query is given below:
            {question}
    """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"context": context, "question": query})

    return response

# Function to Get LLM Data
def get_llm_data(query, llm):
    system = """
        You are a knowledgeable and approachable Computer Science professor with expertise in a wide range of topics.
        Your role is to provide clear, easy, and engaging explanations to help students understand complex concepts.
        When answering:
        - Start with a high-level overview, then dive into details as needed.
        - Use examples, analogies, or step-by-step explanations to clarify ideas.
        - Ensure your answers are accurate, well-structured, and easy to follow.
        - If you don’t know the answer, acknowledge it and suggest ways to explore or research further.
    """

    user = """{query}
    """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"query": query})

    return response

# Function to Get Context Based on Input Query
def get_context(query):
    context = ""

    if use_vector_store:
        with st.spinner(":green[Extracting Data From VectorStore...]"):
            result = "\n\n".join([_.page_content for _ in vector_store.similarity_search(query, k=3)])
            clean_data = clean_rag_data(query, f"Documents Data \n\n{result}", llm)
            context += f"Documents Data: \n\n{clean_data}"

    if use_chat_history:
        with st.spinner(":green[Extracting Data From ChatHistory...]"):
            last_messages = st.session_state.messages[:-1][-3:]
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])
            clean_data = clean_rag_data(query, f"\n\nChat History \n\n{chat_history}", llm)
            context += f"\n\nChat History: \n\n{clean_data}"

    try:
        if use_web:
            with st.spinner(":green[Extracting Data From web...]"):
                search = DuckDuckGoSearchRun()
                clean_data = clean_rag_data(query, search.invoke(query), llm)
                context += f"\n\nWeb Data:\n{clean_data}"
    except:
        pass

    if not use_chat_history:
        with st.spinner(":green[Extracting Data From ChatPPT...]"):
            context += f"\n\n LLM Data {get_llm_data(query, llm)}"

    return context

# Function to Respond to User Based on Query and Context
def respond_to_user(query, context, llm):
    system_prompt = """
    You are an Assistant specialized in Machine Learning (ML) tasks. Your job is to answer the given question based on the following types of context: 

    1. **Web Data**: Information retrieved from web searches.
    2. **Documents Data**: Data extracted from documents (e.g., research papers, reports).
    3. **Chat History**: Previous interactions or discussions in the current session.
    4. **LLM Data**: Insights or completions provided by the language model.

    When answering:
    - Clearly identify the source(s) of the information you are using.
    - If multiple sources are available, integrate them into a cohesive summary.
    - If no relevant context is available, explicitly state that the answer is not known.
    - Ensure your response is clear and easy to understand and remember even for a naive person.
    """
    user_prompt = """Question: {question} 
    Context: {context} """

    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    )

    rag_chain = rag_chain_prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"question": query, "context": context})

    return response

if "messages" not in st.session_state:
    st.session_state.messages = []

if groq_api_key:
    with st.container():
        user_inp = multimodal_chatinput()
    if user_inp:
        question=""
        if user_inp["images"]:
            b64_image=user_inp["images"][0].split(",")[-1]
            image = Image.open(io.BytesIO(base64.b64decode(b64_image)))
            question = img_to_ques(image, user_inp["text"])
            user_inp["text"]=""

        st.session_state.messages.append({"role": "user", "content": question+user_inp["text"]})
        context = get_context(question+user_inp["text"])
        with st.spinner(":green[Combining jhol jhal...]"):
            assistant_response = respond_to_user(question+user_inp["text"], context, llm)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        display_chat_history()
