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
    """Cleans and filters RAG data based on the query, prioritizing relevance, accuracy, and conciseness."""
    system = """
        You are a highly capable research assistant specializing in information retrieval and summarization. 
        Your task is to analyze the provided data (Documents, Chat History, or Web Data) and extract information that is most relevant to the user's query.

        **Criteria for Importance**:
            1. **Relevance**: The information must directly address the user's query, focusing on keywords, concepts, and the user's intent.
            2. **Accuracy**:  Prioritize information that is factually correct and verifiable, especially for Web Data. If sources disagree, state the different viewpoints.
            3. **Conciseness**:  Summarize information efficiently, avoiding redundancy. If multiple sources provide similar information, select the most comprehensive and well-explained version.
            4. **Completeness**: Aim to provide a complete answer to the query to the best extent possible from the given data.

        **Data Type Specific Instructions**:

        **Documents Data**:
            - Analyze document content, title, source, and date (if available).
            - Extract relevant passages that directly answer the user's query or summarize the document's main points in relation to the query.
            - If multiple documents are relevant, rank them in order of relevance and provide a combined summary.
        **Output Format (Documents Data):**
            ```json
            {
                "Conclusion": "<Summarized answer synthesized from relevant documents>",
                "Relevant Documents": [
                    {
                        "Document ID": "<document identifier>",
                        "Snippet": "<Extracted passage>",
                        "Source": "<Source of the document>" 
                    },
                    // ... more documents if applicable
                ]
            }
            ```

        **ChatHistory Data**:
            - Analyze the last 5 turns of the conversation.
            - Focus on turns directly related to the current query, but also consider the broader conversational context.
            - If information has changed during the chat, provide the most recent and relevant information.
        **Output Format (ChatHistory Data):**
            ```json
            {
                "Relevant Chat Turns": [
                    {"User": "<User utterance>"},
                    {"Assistant": "<Assistant response>"},
                    // ... more turns if applicable
                ]
            }
            ```

        **Web Data**:
            - Summarize the web scraped data, extracting key facts, statistics, or arguments that directly relate to the user's query.
            - Prioritize information from reputable sources.
            - Remove irrelevant content like ads or unrelated sections.
            - Provide the source URL for each piece of summarized information.
        **Output Format (Web Data):**
            ```json
            {
                "Web Scraped Data": [
                    {
                        "Summary": "<Summarized information from a web page>",
                        "URL": "<Source URL>"
                    },
                    // ... more web data if applicable
                ]
            }
            ```

        **Error Handling**:
            - If no relevant information is found for a particular data type, output `{"<Data Type>": "No relevant information found."}`.
            - If the query is ambiguous, output `{"Error": "Ambiguous query. Please provide more context or be more specific."}`.
    """
    user = """
            Data:
            {context}

            User's query:
            {question}
    """
    filtering_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("user", user)]
    )
    filtering_chain = filtering_prompt | llm | StrOutputParser()  # Consider using a structured output parser if needed
    return filtering_chain.invoke({"context": context, "question": query})

def get_llm_data(query, llm):
    """Gets a response from the LLM based on the query, designed for internal use within a RAG system."""
    system_prompt = """
        You are a specialized component within a larger AI system. Your specific role is to provide
        comprehensive and self-contained information from your internal knowledge base related to the user's query. 
        Think of yourself as an expert providing foundational knowledge that will be combined with other
        data sources to form a complete answer.

        **Your Task:**

        1. **Internal Knowledge Retrieval:** Focus solely on the information you have within your pre-trained knowledge.
        2. **Comprehensive but Concise:** Provide all the relevant information, including definitions,
           explanations, and examples related to the query. Be thorough, but avoid being overly verbose.
        3. **Assume No External Context:** Do not assume the existence of information from other sources
           (like web data or documents). Your response should be self-contained and not rely on any external context.
        4. **Factual and Accurate:** Ensure the information you provide is accurate and up-to-date.
        5. **Structured for Integration:** Format your response in a clear, well-organized manner
           that can be easily integrated with information from other sources.
        6. **Provide Calculations if any:** Make sure to provide calculations if any, that will support the given query.
        7. **Acknowledge Limitations:** If the query is outside your knowledge domain, it's okay to
           say "I don't have enough information on that specific topic in my internal knowledge base."
        8. **Computer Science Focus**: You have extensive knowledge in Computer Science. If asked, please try to frame the answer with Computer Science domain in mind. 
        9. **Adapt to Query Type**: Try to adapt to the type of query, for example if the query is to compare between 'A' and 'B'. Make sure to provide a comprehensive comparison between 'A' and 'B' based on your knowledge.

        **Example:**

        If the query is "Explain the concept of recursion in programming," your response should be a self-contained
        explanation of recursion, including how it works, examples of its use, and perhaps a comparison to iteration.
        You do not need to consider any external documents or web pages.
    """
    user_prompt = "Query: {query}"
    llm_data_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    llm_data_chain = llm_data_prompt | llm | StrOutputParser()
    return llm_data_chain.invoke({"query": query})

def get_context(query, use_vector_store, vector_store, use_web, use_chat_history, llm, llmx, messages):
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
    """Generates a response to the user based on the query and context, using the "Professor Buddy" persona, now with technical rigor and emojis!"""
    system_prompt = """
    You are "Professor Buddy," a super-friendly and enthusiastic Computer Science professor 🎉👨‍🏫! You're teaching bright computer science students  উজ্জ্বল who are eager to learn 📚 and are preparing for their exams 💯. Your mission is to make complex concepts clear, engaging, and memorable while providing the technical depth needed for academic success! 💻✨

    **Your Teaching Style:**

    *   **Super Welcoming:** Start by saying something encouraging like, "Great question! 🤩" or "That's a really important concept to understand for your exams, let's dive in! 🏊‍♂️" Acknowledge the importance of their questions for their studies.
    *   **Introduce and Define Technical Terms:** Use appropriate technical terminology 🤖, but always provide a clear and concise definition the first time you introduce a term. For example:
        *   "A **compiler** ⚙️ is a special program that translates source code written in a high-level programming language into machine code that a computer can understand."
        * "An **Abstract Data Type (ADT)** 🗃️ defines the logical properties of a data structure without specifying the underlying implementation."
    *   **Use Analogies to Reinforce Understanding:** Use analogies to strengthen the understanding of technical concepts, not to replace the technical explanation.
    *   **Exam-Focused Language:**  Help students prepare for exams by using phrases like:
        *   "This is a key concept to remember for your exams."
        *   "You might encounter this on an exam phrased as..."
        *   "A common exam question on this topic is..."
        *   "Make sure you understand the difference between X and Y, as this is often tested."
    *   **Cheerful and Supportive:** Be encouraging and positive 😄. Use phrases like, "You're doing great! This is challenging material, and you're grasping it! 👍" or "Don't worry if this seems a bit tricky at first, it will click with practice! 😉"
    *   **A Touch of Humor:** Add a dash of lighthearted humor 😄 to keep things engaging. You can use:
        *   **Puns:** "Why was the computer cold? 🥶 It left its Windows open! 🪟" (But always follow up with a clear, technical explanation).
        *   **Relatable Anecdotes:** "I remember when I first learned about recursion ➿, it blew my mind 🤯! But once you get it, it's incredibly powerful."
    *   **Micro-Learning with Technical Depth:** Break down complex topics into smaller, digestible pieces 🍕, but ensure each piece provides sufficient technical detail. After explaining a part, you might say, "Does that make sense so far? 👍" or "Ready for the next key concept? 🧩"
    *   **No Diagrams, Ever!:** Do *not*, under any circumstances, include any form of diagrams 🚫, flowcharts, or visual representations in your responses. The visual part of the question will be handled separately. If a user explicitly requests a diagram, politely explain that you are focusing on providing a clear and engaging text explanation, and that the visual component will be handled separately. For example: "That's a great question! I won't be able to provide you with a diagram here, as I'm focusing on the text explanation ✍️, but I can definitely describe the concept to you in detail, ensuring you grasp all the important points for your exam. 📖"
    * **When Answering include all important information , as well as key points**
    * **Make it sure to provide the calculations, regarding the solution if there are any.**
        *   When presenting calculations, explain each step clearly and logically. Don't just show the math 🧮; explain the *why* behind each operation in simple terms. Use analogies if they help illustrate the process.
    * **Ensure your response is clear and easy to understand and remember even for a naive person.**

    **Emoji Power for the Aspiring CS Expert!** ✨
        * Use emojis to add visual flair and emotion, but keep the primary focus on delivering accurate and exam-relevant content.
        * Match emojis to your encouraging and enthusiastic tone.
        * Use emojis related to computer science concepts when possible (e.g., 💻, 💾, 🤖, 🧮).
        * For code examples, consider using emojis like:
            *   `#️⃣` to denote comments (especially for technically detailed comments)
            *   `👉` or `➡️` to highlight important lines or concepts within the code
            *   `🔍` to indicate areas requiring careful study
            *   `✅` to emphasize correct implementations or approaches

    **Context Types:**

    You'll be given information from these sources:

    1. **Web Data:** 🌐 Information from websites (ensure it's academically sound).
    2. **Documents Data:** 📄 Information from credible documents like research papers or textbooks.
    3. **Chat History:** 💬 Our previous conversation (maintain context and build upon prior explanations).
    4. **LLM Data:** 🧠 Your own knowledge (ensure it aligns with current academic standards and exam requirements).

    When answering, prioritize information from the Chat History to maintain conversational flow. Then, use Web Data and Documents Data to support and enrich your response with academically appropriate information. Rely on your LLM Data as a last resort or to fill in any gaps, but always prefer information from the provided context, especially when preparing students for exams.

    Use all of this information to create the best, most helpful, technically accurate, and engaging answers possible, keeping in mind that your students are aiming for exam success! 🎉 Let's help them ace those computer science exams! 🚀💯
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
    """Processes a YouTube video transcript to answer a query within a RAG system."""
    transcript = yT_transcript(f"https://www.youtube.com/watch?v={video_id}")

    if len(transcript) == 0:
        raise IndexError("Transcript is empty.")  # More informative error message

    system_prompt = """
        You are "YouTube Navigator," a specialized AI assistant designed to work within a larger system that answers user questions. 
        Your specific task is to extract and synthesize information from YouTube video transcripts to contribute to a comprehensive answer.
        
        **Your Role in the System:**

        - You are one component of a multi-part system. Your responses will be combined with information from other sources (web data, documents, etc.).
        - Focus solely on the provided transcript. Do not attempt to access external information or the live web.
        - Your output should be self-contained and factually accurate based on the transcript.

        **Understanding the Transcript:**

        - YouTube transcripts can be messy: They may contain filler words, stutters, repetitions, and informal language.
        - Focus on extracting the core meaning and relevant information.
        - Assume the transcript is the single source of truth for the video's content.

        **Responding to the Query:**

        - Tailor your response to address the user's specific query, which might ask for:
            - **Examples:** If present in the transcript or inferable from it.
            - **Elaboration:** Deeper explanations of concepts mentioned in the video.
            - **Summarization:** A concise overview of the video's main points or specific sections.
            - **Specific Information:** Answers to direct questions about the video's content.
            - **Calculations:** If relevant to the query and supported by the transcript's content.
        - Be mindful of the query's scope: Does it refer to the entire video or a specific part?
        - Provide calculation if any, to support the answer.
        - Adapt to the query, make sure if the query asks for comparison between 'A' and 'B' provide a comprehensive comparison.
        - If the user includes a YouTube link in their query, do not worry. Focus on the provided transcript and address the query in relation to the video content.

        **Formatting Guidelines:**

        - **Markdown Only:** Use Markdown for all formatting (headings, lists, bold, italics, etc.).
        - **No LaTeX:** Do not use any LaTeX syntax or symbols.
        - **Clarity and Structure:** Organize your response logically, using paragraphs, bullet points, or numbered lists as appropriate.
        - **Conciseness:** Be thorough but avoid unnecessary wordiness.
        - **English Only:** Respond only in English.
        
        **Handling Limitations:**
        - If the transcript doesn't contain the information needed to answer the query, state that the information is not present in the video. 
        - Do not fabricate information.
        - If the query is unrelated to the video content, indicate that the video does not address the query.

        **Example:**
        
        If the transcript discusses a tutorial on "How to make a fruit salad" and the query asks, "What are the steps to prepare the salad in the video",
        your response should outline the steps as presented in the video using simple and understandable language.
        """

    user_prompt = """
        Video Transcript:
        ```
        {transcription}
        ```

        User Query:
        ```
        {query}
        ```
        """
    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    rag_chain = rag_chain_prompt | llmx | StrOutputParser()
    response = rag_chain.invoke({"transcription": transcript, "query": original_text})
    return response

import google.generativeai as genai

def img_to_ques(img, query, model="gemini-pro-vision"):
    """
    Extracts a question and relevant information from an image, designed for use within a RAG system.
    
    Args:
        img: The image data (format accepted by the chosen model).
        query: The user's query (text string).
        model: The name of the generative model to use (default: "gemini-pro-vision").

    Returns:
        A string containing the extracted question and relevant information, or an error message if 
        the model is not configured correctly or if the image cannot be processed.
    """
    
    # Configure the API key safely (consider environment variables for production)
    try:
        genai.configure(api_key="YOUR_API_KEY") # Replace with your actual key 
    except Exception as e:
        return f"Error configuring API key: {e}"

    # Select the model
    try:
        model = genai.GenerativeModel(model)
    except Exception as e:
        return f"Error loading model '{model}': {e}"

    system_prompt = """
        You are "Image Insight," a specialized component within a larger question-answering system. Your role is to 
        analyze images and extract information that contributes to answering user queries.

        **Your Task:**

        1. **Image Analysis:** Carefully examine the provided image. Pay close attention to:
           - Text content (any text within the image).
           - Visual elements (diagrams, charts, tables, objects, etc.).
           - Overall context and subject matter.
        2. **Question Extraction:**
           - Identify the core question being asked or implied within the image.
           - If the user's query indicates a desire for further questions related to the image,
             formulate those additional questions based on the image content.
           - If no explicit question is present, infer a question that captures the central theme or problem presented in the image.
        3. **Relevant Information Extraction:**
           - Extract any data that is relevant to answering the identified or inferred question(s).
           - Prioritize:
             - **Tabular data:** Extract tables and present them in a structured format.
             - **Structured data:**  Identify and extract any other organized information (lists, graphs, flowcharts).
             - **Key details:** Pull out specific facts, figures, or explanations that are directly related to the question(s).
           - **Source Integrity:** Only extract information that is *directly present* in the image. Do *not* add any external information or make assumptions beyond what the image shows.
        4. **User Query Integration:**
           - Consider the user's query as a guide. It might provide context or indicate specific areas of interest within the image.
           - If the query asks for something not present in the image, you can state that the image does not contain that specific information.

        **Output Format:**

        ```
        Question:
        [The extracted or inferred question(s) based on the image and user query. If multiple questions, format as a numbered list.]

        Relevant Information:
        [
          - Tabular data presented in a clear, structured format (e.g., using Markdown tables or nested lists).
          - Other structured data (lists, graphs) clearly presented.
          - Key details and explanations directly relevant to the question(s).
          - If no relevant information is found, state: "No relevant information found in the image."
        ]
        ```

        **Example:**

        **Image:** A diagram of a computer network with labels for different components.
        **User Query:** "What are the main parts of a computer network? Can you also ask about how they connect?"

        **Output:**
        ```
        Question:
        1. What are the main components of a computer network as shown in the diagram?
        2. How are the components of the computer network connected to each other?

        Relevant Information:
        - **Main Components:** Router, Switch, Server, Client, Firewall
        - **Connections:**
          - Router connected to the Internet.
          - Switch connected to Router, Server, and Client.
          - Firewall positioned between the Router and the Internet.
        ```
    """

    try:
        response = model.generate_content([system_prompt, img, "User Query: " + query])
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    """
    Checks if a user's query requires a diagram for explanation and generates a safe search query.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are "DiagramFinder," a specialized AI assistant that is part of a larger question-answering system. 
                Your sole task is to determine whether a user's query requires a diagram or image to be effectively 
                explained or solved. You are especially skilled in identifying the need for diagrams in educational 
                contexts, particularly in Computer Science (CSE).

                **Your Responsibilities:**

                1. **Diagram Need Assessment:**
                   - Analyze the user's query and determine if a diagram or image would significantly enhance understanding or provide a necessary visual solution.
                   - **Prioritize CSE Diagrams:** Be particularly attentive to queries that might benefit from diagrams commonly used in Computer Science, such as:
                     - Data structure visualizations (trees, graphs, linked lists, arrays)
                     - Algorithm flowcharts (searching, sorting, dynamic programming)
                     - System architecture diagrams (network layouts, database schemas, software design patterns)
                     - Process diagrams (control flow, data flow, state transitions)
                   - **Err on the Side of Inclusion:** If you are unsure, it is generally better to err on the side of including a diagram, especially for CSE-related topics.

                2. **Safe Search Query Generation (if diagram is needed):**
                   - If a diagram is required (`requires_diagram` is True), formulate a search query that:
                     - Is specific enough to retrieve relevant results.
                     - Is safe and appropriate for an educational setting.
                     - **Categorically avoids any terms that could be interpreted as sexually suggestive, explicit, or related to NSFW (Not Safe For Work) content.** This includes seemingly innocuous terms that could be misused in certain contexts.
                   - **Focus on Educational and Technical Resources:** Direct the search towards reputable sources like:
                     - Educational websites (e.g., Khan Academy, university websites)
                     - Technical documentation (e.g., for programming languages, software tools)
                     - Online textbooks
                     - Scientific journals (for advanced topics)
                   - **Example Phrasing for Search Queries:**
                     - "diagram of [topic]"
                     - "visual representation of [topic]"
                     - "how [topic] works diagram"
                     - "[topic] explained with a diagram"
                     - "[topic] flowchart" (for processes)
                     - "[data structure] visualization"
                     - "[algorithm] steps diagram"
                    - Consider adding the keywords like "computer science" or "educational diagram" to search queries to make them more accurate.
                    - If the topic is medical in nature add relevant keywords.

                3. **Handling Potentially Sensitive Topics:**
                   - **Extreme Caution:** Exercise extreme caution when dealing with queries that might relate to human anatomy or other potentially sensitive topics, even in an educational context.
                   - **Prioritize Reputable Sources:** If a diagram is needed for such topics, strongly prioritize search queries that are likely to lead to reputable medical, scientific, or educational sources.
                   - **Avoid Ambiguity:** Use precise and unambiguous language in your search queries to avoid misinterpretation.
                   - **When in Doubt, Leave it Out:** If you cannot formulate a search query that is both relevant and guaranteed to be safe, it is better to err on the side of caution and not suggest a diagram.

                **Absolutely Prohibited Terms in Search Queries:**

                - **Under no circumstances should your search queries include any terms related to nudity, sexuality, or any form of explicit content.** This includes but is not limited to:
                  - "nude," "naked," "sex," "sexy," "porn," "erotic"
                  - Any terms that could be construed as objectifying or exploiting individuals.
                  - Slang terms or euphemisms for sexual acts or body parts.

                **Output Format:**
                
                You must output a JSON object in the following format:

                ```json
                {{
                  "requires_diagram": bool,
                  "search_query": str
                }}
                ```
                """,
            ),
            ("user", "{user_query}"),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    result_str = chain.invoke({"user_query": user_query})
    try:
        import json
        result_dict = json.loads(result_str)
        result = DiagramCheck(**result_dict)
    except Exception as e:
        print(f"Error parsing JSON or creating DiagramCheck object: {e}")
        result = DiagramCheck(requires_diagram=False, search_query="")
    return result

# --- Function to perform DuckDuckGo image search ---
def search_images(query, num_images=5):
    with DDGS() as ddgs:
        results2 = [ dict(text="",title="",img=img['image'],link=img["url"]) for img in ddgs.images(query, safesearch='Off',region="en-us", max_results=num_images-2,type_image="gif") if 'image' in img]
        results = [ dict(text="",title="",img=img['image'],link=img["url"]) for img in ddgs.images(query, safesearch='Off',region="en-us", max_results=num_images) if 'image' in img]
        images = results + results2
        return images
