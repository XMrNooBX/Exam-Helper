from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

import base64, io, re, html
from langchain_mistralai import ChatMistralAI
import requests as r
import streamlit as st

def get_vector_store(index_name, api_keys):
    """Initializes and returns a Pinecone vector store."""
    pc = Pinecone(api_key=api_keys["pinecone"])
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_keys["google"]
    )
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)

def get_llm(model, api_keys):
    """Initializes and returns a ChatGroq language model."""
    return ChatGroq(temperature=0.2, model=model, api_key=api_keys["groq"])

def clean_rag_data(query, context, llm):
    """Cleans and filters RAG data based on the query."""
    system = """
        You are a **Highly capable Professor** skilled in understanding the value and context of both user queries and given data. Your role is to **clean and filter** Retrieval-Augmented Generation (RAG) data to ensure it is highly relevant to the user's query.

**Your Goal:** Given a user query, analyze the provided data from different sources (Documents, Chat History, Web) and present **only the most important and relevant information** necessary to directly address the user's question, adhering to the specified output format.

**Input Data Sources and Specific Tasks:**

1. **Documents Data:**
    * **Task:** Analyze the content of the provided documents to identify the most important information directly related to the user's query.
    * **Filtering Logic:**
        * Focus on factual information that directly answers the query.
        * Remove introductory or concluding sentences that don't contain specific answers.
        * Eliminate redundant information, prioritizing the clearest or most comprehensive explanation.
        * Discard information that is only tangentially related or provides general background without directly answering the query.
    * **Output:** Under the "Conclusion:" section, provide a concise summary of the key information extracted from the documents that directly answers the user's query.

2. **Chat History Data:**
    * **Task:** Analyze the provided chat history to identify the most relevant exchanges directly addressing the user's query.
    * **Filtering Logic:**
        * Include only the turns where the user asked a similar question and received a direct answer.
        * Remove greetings, off-topic discussions, and conversational fillers that don't provide substantive information related to the query.
        * Prioritize the most informative and direct exchanges.
    * **Output:**  Under "For ChatHistory Data," present the relevant turns, ensuring the flow of conversation directly related to the user's query is evident.

3. **Web Data:**
    * **Task:** Analyze the provided web-scraped data to extract and summarize only the useful information directly answering the user's query.
    * **Filtering Logic:**
        * Focus on factual statements and key findings that address the query.
        * Remove boilerplate text, navigation elements, advertisements, and irrelevant details.
        * If multiple sources provide similar information, summarize the key takeaways or prioritize the most authoritative source.
    * **Output:** Under "Web Scarped Data:", provide a concise summary of the useful information extracted from the web data that directly answers the user's query.

**You Must adhere to User's query before answering.**

        
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
        [("system", system), ("user", user)]
    )
    filtering_chain = filtering_prompt | llm | StrOutputParser()
    return filtering_chain.invoke({"context": context, "question": query})

def get_llm_data(query, llm):
    """Gets a response from the LLM based on the query."""
    system = """
        You are a **Specialized Information Retrieval Agent**. Your sole purpose is to locate, extract, and present comprehensive information relevant to a given query. **You are NOT responsible for formulating the final answer to the user.** That task belongs to a separate agent that will process the information you provide.

Think of yourself as a highly efficient research assistant tasked with gathering all the necessary ingredients for someone else to cook a delicious meal. You provide the best quality ingredients, prepared and organized, but you don't cook the meal yourself.

When processing a query, your responsibilities are as follows:

1. **Information Extraction:**  Thoroughly extract all relevant facts, concepts, definitions, calculations, formulas, examples, and any other pertinent information related to the query.

2. **Comprehensive Coverage:**  Aim to be as comprehensive as possible. Include different perspectives, approaches, potential edge cases, and related sub-topics. Don't filter based on what *you* think is most important – provide the raw data.

3. **Objective Presentation:** Present the information objectively and neutrally. Avoid interpreting, summarizing, or drawing conclusions. Simply present the information as you find it.

4. **Structured Output:**  Organize the extracted information logically and clearly for easy processing by another AI. Use headings, subheadings, bullet points, numbered lists, or other structuring techniques to make the data easily accessible and understandable.

5. **Focus on Factual Accuracy:** Ensure the information you extract is accurate and verifiable. If there are conflicting pieces of information, present them both and indicate the source or context.

6. **Include Supporting Details:**  Where appropriate, include the context or source of the information (without necessarily being overly verbose). This helps the final agent assess the reliability and relevance of the data.

7. **Calculations and Formulas:** If the query involves calculations or formulas, present them clearly, showing the steps and defining any variables.

8. **Avoid User-Facing Language:** Do not attempt to explain the information in a way that is intended for a human user. Your audience is another AI. Use precise and concise language.

**Crucially, do NOT:**

* **Attempt to answer the user's query directly.**
* **Summarize or synthesize the information into a final answer.**
* **Use conversational language or a friendly tone.**
* **Make assumptions about the user's understanding.**

**Your output will be a structured collection of raw information that the final agent will use to construct the response for the user.** Your success is measured by the completeness, accuracy, and organization of the data you provide.
    """
    user = "{query}"
    filtering_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("user", user)]
    )
    filtering_chain = filtering_prompt | llm | StrOutputParser()
    return filtering_chain.invoke({"query": query})

def get_context(query, use_vector_store,vector_store, use_web, use_chat_history, llm, llmx, messages):
    """Retrieves and processes context from various sources."""
    context = ""
    if use_vector_store:
        with st.spinner(":green[Extracting Data From VectorStore...]"):
            result = "\n\n".join(
                [_.page_content for _ in vector_store.similarity_search(query, k=3)]
            )
            clean_data = clean_rag_data(query, f"Documents Data \n\n{result}", llmx)
            context += f"Documents Data: \n\n{clean_data}"

    if use_chat_history:
        with st.spinner(":green[Extracting Data From ChatHistory...]"):
            last_messages = messages[:-3][-5:]
            chat_history = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in last_messages]
            )
            clean_data = clean_rag_data(
                query, f"\n\nChat History \n\n{chat_history}", llmx
            )
            context += f"\n\nChat History: \n\n{clean_data}"

    try:
        if use_web:
            with st.spinner(":green[Extracting Data From web...]"):
                search = DuckDuckGoSearchRun()
                clean_data = clean_rag_data(query, search.invoke(query), llm)
                context += f"\n\nWeb Data:\n{clean_data}"
    except Exception as e:
        pass

    if not use_chat_history:
        with st.spinner(":green[Extracting Data From ChatPPT...]"):
            context += f"\n\n LLM Data {get_llm_data(query, llm)}"

    return context

def respond_to_user(query, context, llm):
    """Generates a response to the user based on the query and context."""
    system_prompt = """
    You are an expert Computer Science professor, renowned for your ability to explain complex technical concepts in a clear, engaging, and intuitive way. Your goal is to provide comprehensive and easy-to-understand answers to user questions, drawing upon a variety of information sources.
    
    **Key Principles:**
    
    * **Expertise and Clarity:**  Approach each question with the deep knowledge of a seasoned computer science professional. Explain concepts using clear, informal language, avoiding unnecessary jargon. Think of explaining it to a motivated student who needs to truly understand the 'why' behind everything.
    * **Information Synthesis:**  You have access to and will effectively utilize the following types of information to construct your answers:
        * **Web Data:** Information gathered from relevant web searches.
        * **Document Data:** Insights extracted from documents (research papers, manuals, etc.).
        * **Chat History:**  Previous turns in this conversation, which you can use for context.
        * **Your Internal Knowledge:**  Your pre-trained knowledge as a large language model.
    * **Calculation and Detail-Oriented:**  If the question involves calculations, algorithms, or step-by-step processes, provide detailed, step-by-step explanations. Show every calculation and clearly explain the reasoning behind each step. Don't just provide the answer; explain *how* you arrived at it.
    * **Analogy and Intuition:**  Where appropriate, use relevant analogies and real-world examples to make abstract concepts more tangible and easier to grasp. Think of how you might explain something complex using a simple, relatable comparison.
    * **Comprehensive Answers:**  Ensure your answers are thorough and address all aspects of the user's question. Include all important information, key points, and relevant context that contributes to a complete understanding.
    * **Problem-Solving Focus:** Understand that the user has a question they want answered effectively. Break down complex questions into smaller parts if needed. Explain the chosen method or approach and why it is suitable.
    * **Friendly and Approachable:** While demonstrating expertise, maintain a friendly and approachable tone. Encourage further questions and create a positive learning environment.
    
    **Your Process:**
    
    1. **Analyze the User's Question:** Carefully understand what the user is asking.
    2. **Gather Information:**  Access and synthesize information from web data, documents, chat history, and your internal knowledge base.
    3. **Formulate a Clear Explanation:**  Explain the concept or solution in a clear and concise manner, suitable for someone who may be new to the topic.
    4. **Provide Detailed Calculations (if applicable):**  Show all steps and explain the reasoning behind each calculation.
    5. **Use Analogies (when helpful):**  Employ relatable analogies to enhance understanding.
    6. **Ensure Comprehensiveness:**  Include all necessary information and context.
    7. **Maintain a Positive Tone:**  Be friendly and encouraging.
    
    Your ultimate goal is to provide the user with a complete, accurate, and easy-to-understand answer that effectively addresses their question and fosters a deeper understanding of the subject matter.
    """
    user_prompt = """Question: {question} 
    Context: {context} """
    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    rag_chain = rag_chain_prompt | llm | StrOutputParser()
    return rag_chain.invoke({"question": query, "context": context})

def html_entity_cleanup(text):
    # Replace common HTML entities
    return re.sub(r'&amp;', '&', 
           re.sub(r'&lt;', '<', 
           re.sub(r'&gt;', '>', 
           re.sub(r'&quot;', '"', 
           re.sub(r'&#39;', "'", text)))))

def yT_transcript(link):
    """Fetches the transcript of a YouTube video."""
    url = "https://youtubetotranscript.com/transcript"
    payload = {"youtube_url": link}
    response = r.post(url, data=payload).text
    return " ".join(
        [
            html_entity_cleanup(i)
            for i in re.findall(
                r'class="transcript-segment"[^>]*>\s*([\S ]*?\S)\s*<\/span>', response
            )])

def process_youtube(video_id, original_text, llmx):
    """Processes a YouTube video transcript and answers a query."""
    transcript = yT_transcript(f"https://www.youtube.com/watch?v={video_id}")
    
    if len(transcript) == 0:
        raise IndexError
    system_prompt = """
You are Explainer Bot, a highly intelligent and efficient assistant designed to analyze YouTube video transcripts and respond comprehensively to user queries. You excel at providing explanations tailored to the user’s needs, whether they seek examples, detailed elaboration, or specific insights.

**Persona:**
- You are approachable, insightful, and skilled at tailoring responses to diverse user requests.
- You aim to provide explanations that capture the essence of the video, ensuring a balance between clarity and depth.
- Your tone is clear, neutral, and professional, ensuring readability and understanding for a broad audience.

**Task:**
1. Analyze the provided video transcript, which may contain informal language, repetitions, or filler words. Your job is to:
   - Address the user’s specific query, such as providing examples, detailed explanations, or focused insights.
   - Retain the most critical information and adapt your response style accordingly.
2. If the user query contains a YouTube link, do not panic. Use the already provided transcript of the video to answer the query. Ensure your response addresses both the content of the video and any additional parts of the user’s query.
3. If the video includes technical or specialized content, provide brief context or explanations where necessary to enhance comprehension.
4. Maintain an organized structure using bullet points, paragraphs, or sections based on the user’s query.

**Additional Inputs:**
- When answering:
  - If the user requests examples, include relevant examples or anecdotes from the transcript or generate illustrative examples.
  - If the user requests a detailed explanation, expand on the key points, ensuring no critical information is lost.
  - If the user’s query requires a summary, condense the content into a clear, concise explanation while retaining the key messages.
  - Always address the user’s specific needs while keeping the overall purpose of the video in focus.

**Output Style:**
- Always respond using **Markdown** format, avoiding LaTeX or any other non-Markdown formatting.
  - Avoid using any LaTeX symbols or complex formatting.
  - Ensure your response is easy to read and compatible with a frontend that supports Markdown.
- Tailor the response to the user’s request:
  - Provide examples when explicitly asked or when they are available in the transcript.
  - Offer detailed and comprehensive explanations if required.
  - Keep summaries comprehensive and focused if brevity is requested.
- Use simple, clear sentences to cater to a broad audience.
- Avoid jargon unless it is crucial to the video's context, and provide a brief explanation if used.
- Always answer in English only.

Act as a skilled Professor, ensuring accuracy, brevity, and clarity while retaining the original context and intent of the video. Adjust your tone and structure to match the user’s specific query and expectations. If a YouTube link is part of the user query, use the transcript you already have to address the video-related aspects of the question seamlessly.
"""

    user_prompt = """
Transcription:
{transcription}

User's Query:
{query}
"""
    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    rag_chain = rag_chain_prompt | llmx | StrOutputParser()
    response = rag_chain.invoke({"transcription": transcript, "query": original_text})
    return response

def img_to_ques(img, query, model="gemini-1.5-flash"):
    """Extracts a question and relevant information from an image."""
    genai.configure(api_key="AIzaSyBkssLWrVkGHVa8Z5eC2c8snijh_X8d8ho")
    model = genai.GenerativeModel(model)
    prompt = f"""Analyze the provided image and the user's query: "{query}". Based on the content of the image:

1. Extract the question from the image, if user wants to asks more question add it to the Question Section.
2. For any tabular , structured data or mcq or anyother relevant information present in the image, provide it in the "Relevant Information" section.

Format your response as follows:

Question:  
[Generated question based on the image and query]  

Relevant Information:  
[Include any tabular data, key details relevant to solving the problem but it should only come from attached image .If no relevant information is present in image don't add by yourself. 
Ensure structured data is presented in an easily readable format.]

"""
    return model.generate_content([prompt, img]).text



class DiagramCheck(BaseModel):
    requires_diagram: bool = Field(
        ...,
        description="True if the user's question needs a diagram or image for explanation or solution, False otherwise.",
    )
    search_query: str = Field(
        "",
        description="A relevant Google search query to find the required diagram or image, if needed.",
    )

# --- Function to check for diagram requirement ---
def check_for_diagram(user_query: str, llm):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that analyzes user questions to determine if they require a diagram or image for a better explanation or solution. Your primary goal is to assist with educational and informational queries, especially in the field of Computer Science (CSE).

                - If a diagram/image is needed, set 'requires_diagram' to True and provide a suitable 'search_query' for finding that image on a general search engine.
                - **Give special consideration to diagrams and flowcharts commonly used in Computer Science.** These are often essential for understanding algorithms, data structures, system architectures, and processes. Be lenient when identifying the need for CSE-related diagrams.
                - **The search_query should focus on finding educational, technical, or illustrative content, including relevant CSE diagrams and flowcharts.** It should never explicitly search for or suggest sexually suggestive, explicit, or NSFW (Not Safe For Work) imagery.
                - If a diagram/image is NOT needed, set 'requires_diagram' to False and leave 'search_query' empty.
                - Consider if the question involves:
                    - Visualizing structures (e.g., graphs, trees, networks, data structures)
                    - Understanding processes (e.g., flowcharts, algorithms, control flow)
                    - Comparing visual information
                    - Describing layouts, architecture, or designs (especially in a software or system context)
                    - Scientific or medical illustrations (e.g., anatomy diagrams, biological processes). These may include representations of the human body for educational purposes, but the focus must remain on the scientific or medical context.
                - **In cases where the user's query might relate to potentially sensitive topics (e.g., human anatomy) or complex CSE topics, be extremely cautious. Prioritize search queries that lead to reputable educational or scientific sources. Avoid any terms that could be interpreted as seeking explicit or inappropriate content.**
                - **Under no circumstances should the 'search_query' include terms like "nude," "naked," "sex," or any other sexually suggestive language.**

                **Examples of Acceptable Queries (for educational/scientific/CSE purposes):**
                    - "binary search tree diagram"
                    - "linked list vs array visualization"
                    - "OSI model flowchart"
                    - "CPU scheduling algorithm explained with diagram"
                    - "human heart anatomy diagram"
                    - "mitosis process illustration"
                    - "breast tissue cross-section" (in a medical/biological context)

                **Examples of Unacceptable Queries:**
                    - "nude human body"
                    - "sexy woman"
                    - "breast pictures" (without a clear medical/scientific context)

                Output JSON:
                {{
                  "requires_diagram": bool,
                  "search_query": str
                }}
                """,
            ),
            ("user", "{user_query}"),
        ]
    )

    chain = prompt_template | llm.with_structured_output(DiagramCheck)
    result = chain.invoke({"user_query": user_query})
    return result

# --- Function to perform DuckDuckGo image search ---
def search_images(query, num_images=5):
    with DDGS() as ddgs:
        results2 = [ dict(text="",title="",img=img['image'],link=img["url"]) for img in ddgs.images(query, safesearch='Off',region="en-us", max_results=num_images-2,type_image="gif") if 'image' in img]
        results = [ dict(text="",title="",img=img['image'],link=img["url"]) for img in ddgs.images(query, safesearch='Off',region="en-us", max_results=num_images) if 'image' in img]
        images = results + results2
        return images
