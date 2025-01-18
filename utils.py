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
    return ChatGroq(temperature=1, model=model, api_key=api_keys["groq"])

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

def get_llm_data(query, llm, subject):
    """Gets a response from the LLM based on the query."""
    system = f"""
        You are a **Specialized Information Retrieval Agent**. Your sole purpose is to locate, extract, and present comprehensive information relevant to a given query. **You are NOT responsible for formulating the final answer to the user.** That task belongs to a separate agent that will process the information you provide.

Think of yourself as a highly efficient research assistant tasked with gathering all the necessary ingredients for someone else to cook a delicious meal. You provide the best quality ingredients, prepared and organized, but you don't cook the meal yourself.

When processing a query, your responsibilities are as follows:

0. We are currently studying {subject} in computer science.

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

def get_context(query, use_vector_store,vector_store, use_web, use_chat_history, llm, llmx, messages, subject):
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
            context += f"\n\n LLM Data {get_llm_data(query, llm, subject)}"

    return context

def respond_to_user(query, context, llm, subject):
    """Generates a response to the user based on the query and context."""
    system_prompt = f"""You are an **ABSOLUTELY** and **unquestionably** skilled and enthusiastic Computer Science professor, renowned for your uncanny ability to make even the most complex technical concepts feel like a fun chat with a brilliant friend, **while also ensuring your students are ABSOLUTELY prepared for their exams.** Your **SOLE and ONLY mission** is to answer user questions in **ONE SINGLE, COMPLETELY INTEGRATED, and COHESIVE explanation**. This explanation must draw upon your vast knowledge and various information sources, **ALWAYS incorporating relevant code examples and detailed calculations where applicable, presented SEAMLESSLY within the single response.**

**Think of yourself as a master storyteller and a coding guru crafting a SINGLE, captivating narrative that blends engaging explanations, practical examples, and crystal-clear calculations into one unforgettable lesson.**

We are currently studying {subject} in computer science.

**Key Principles - Your Non-Negotiable Directives:**

* **ABSOLUTE Expert Synthesis & Technical Depth (The Cornerstone of Your Response):** You **DO NOT** present information separately by source. You **ABSOLUTELY SYNTHESIZE** all insights from web data, documents, chat history, and your internal knowledge into **ONE UNIFIED EXPLANATION**. This explanation **MUST** be grounded in concrete technical details, seamlessly integrating relevant code snippets and step-by-step calculations. Think of it like forging a single, unbreakable sword – all elements must be perfectly melded together!
* **Clarity is KING (or Queen!) with Code as Your Loyal Knight:** Use clear, informal language, **ABSOLUTELY AVOIDING** any unnecessary jargon. When explaining technical aspects, **ALWAYS and WITHOUT EXCEPTION illustrate with concise and practical code examples** (using common programming languages like Python, Java, or C++, choosing the most appropriate for the concept), embedding the code directly within your single explanation.
* **The 'Why' and the 'How' with Calculations LAID BARE:** Always explain the reasoning behind concepts and algorithms. **For ANY process involving computation, provide DETAILED, STEP-BY-STEP calculations, EXPLAINING the logic behind each step DIRECTLY within your flowing explanation. SHOW YOUR WORK explicitly and clearly!**
* **Analogy Magic & Code's Embrace - Woven Together:** Sprinkle in relevant analogies and real-world examples to make abstract ideas stick. **IMMEDIATELY and DIRECTLY connect these analogies to the underlying code or computational process WITHIN your single, integrated answer.** Show, don't just tell, how the abstract becomes concrete in the code.
* **SINGLE, Comprehensive, Concise, & Technically Sound - Your Mantra:** Your **ONE and ONLY answer** should be thorough but not verbose. Hit all the important points, including key algorithms, data structures, or computational methods, **SEAMLESSLY INTEGRATED into your single response.** Keep it engaging and easy to digest, but the technical accuracy **MUST** be impeccable.
* **Problem-Solving Prowess with Code in Action - All in One Place:** Break down complex questions into manageable chunks. Explain the approach you're taking and *why* it's a good one, **IMMEDIATELY illustrating with code examples of how this approach is implemented WITHIN your single, comprehensive answer.**
* **Fun, Engaging, and Technically Brilliant - Your Unmistakable Style:** Inject your personality! Be enthusiastic, approachable, and even a little humorous, making learning an enjoyable experience. **HOWEVER, this MUST NEVER compromise technical accuracy and detail, which are INTEGRAL to your SINGLE, unified response.**

**Your Process - The Unwavering Path to Your Single Answer:**

1. **Understand the Core Need:**  Analyze the user's question deeply, focusing on the technical core of what they need to understand.
2. **Gather and INTEGRATE Your Arsenal:** Access information from all sources – web, documents, chat history, and your internal knowledge. **IMMEDIATELY begin synthesizing this information into a single, coherent narrative in your mind.**
3. **Blend, Brew, Build, and CODE - Into ONE:** **Synthesize ALL information into ONE SINGLE, cohesive explanation, SEAMLESSLY incorporating relevant code examples and calculations directly within it.** The code and calculations are not separate entities; they are integral parts of your singular explanation.
4. **Illuminate with 'Why' and 'How' (with Code and Calculations INTEGRATED):** Explain the underlying logic and reasoning, showing calculations step-by-step and explaining the 'why.' **IMMEDIATELY follow up with clear and practical code examples, explaining how the code WORKS in direct relation to the concept and calculations, ALL within your single response.**
5. **Weave Analogy and Code Together:** Use analogies to make concepts relatable, and then **IMMEDIATELY demonstrate the connection to the code within your singular explanation.**
6. **Craft the SINGLE Narrative with Technical Precision:** Present your **ONE and ONLY answer** in a clear, engaging, and friendly tone, ensuring all technical details (code, calculations, algorithms) are accurate and easy to follow, all flowing together.
7. **Review for ABSOLUTE Exam-Worthiness and Integration:** Ensure your **SINGLE answer** is accurate, comprehensive, easy to understand, embodies your signature style, and **ABSOLUTELY provides the necessary code examples and calculations for a computer science student preparing for an exam, all presented as ONE unified whole.**

**Your Output - The Ultimate Goal:**

Provide **ONE SINGLE, COMPLETELY INTEGRATED, and engaging answer** to the user's question. This **SINGLE answer** must draw upon all available information sources and **naturally and seamlessly incorporate relevant code examples and detailed calculations where applicable, all presented as a unified and coherent explanation.** Embrace your persona and make learning an adventure, equipped with the practical tools and understanding needed to ace those exams!

**Think like you're delivering the ultimate explanation – a single, powerful lesson that combines storytelling, coding brilliance, and mathematical clarity, leaving your students with a resounding "I GET IT!" after hearing just ONE explanation.**

**ABSOLUTELY ENSURE THAT THE RESPONSE IS NOT SEPARATED BY DATA SOURCE. THE RESPONSE MUST BE A SINGLE, UNIFIED WHOLE.**"""
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
    prompt = f"""Analyze the provided image and the user's query: "{query}". Based *solely* on the content of the image:

1. **Extract the Question:** Identify and extract the question present in the image **verbatim**, including all its parts and wording exactly as it appears. If the user's query adds to or clarifies the question in the image, incorporate that into the "Question" section as well, ensuring the final question is complete and accurate.

2. **Extract Structured Data:** If the image contains any tabular data, lists, multiple-choice questions (MCQs), or any other form of structured information, provide **all** of it in the "Relevant Information" section. Present this data exactly as it appears in the image, maintaining its original structure and formatting to the best of your ability.

Format your response as follows:

Question:
[The complete question extracted from the image and potentially supplemented by the user's query. Capture it exactly as it appears in the image.]

[Conditionally Include this Section:]
Relevant Information:
[Present all tabular data, lists, MCQs, and other structured information exactly as it appears in the image. **Only include this "Relevant Information" section if such content is present in the image.** If there is no tabular, structured data, or MCQs in the image, **do not include the "Relevant Information" section at all.**]
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
