from dataclasses import dataclass
import streamlit as st
import chromadb
import time
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from prompts.prompts import Template
from typing import Literal
from langchain_community.vectorstores import Chroma
from langchain.retrievers import MergerRetriever
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
# api_key = os.getenv("OPENAI_API_KEY")
api_key = st.secrets['OPENAI_API_KEY']

def speech_to_text(audiofile):
    client = OpenAI(api_key=api_key)
    audio_file= open(audiofile, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcription.text
def text_to_speech(speech_file_path,chat_format):
    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
    model="tts-1",
    voice="echo",
    input=chat_format
    )
    response.stream_to_file(speech_file_path)

@dataclass
class Message:
    """Class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str
def jd_retrieval(jd):
    """create embeddings for job description"""
    embeddings = OpenAIEmbeddings()
    if "resumeEmbeddings" in st.session_state and "resumeTimestamp" in st.session_state:
        # Check if the embeddings are not too old (e.g., 1 hour)
        if time.time() - st.session_state.resumeTimestamp < 3600:
            return st.session_state.resumeEmbeddings
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
    jd_splitted = text_splitter.split_text(jd)
    
    chroma_jd = Chroma.from_texts(
        jd_splitted,embeddings,
        collection_metadata={"hnsw:space": "cosine"} # l2 is the default(persist_directory="vector_storage/resume_store")
    )
    retriever_jd = chroma_jd.as_retriever(search_type = "similarity", search_kwargs = {"k":1})
     # Cache the embeddings and timestamp
    st.session_state.resumeEmbeddings = retriever_jd
    st.session_state.resumeTimestamp = time.time()
    return retriever_jd


# passing resume then extract its embeddings and return into its retrieval format.
def resume_retrieval(resume):
    '''Create embeddings for the resume'''
    if "resumeEmbeddings" in st.session_state and "resumeTimestamp" in st.session_state:
        # Check if the embeddings are not too old (e.g., 1 hour)
        if time.time() - st.session_state.resumeTimestamp < 3600:
            return st.session_state.resumeEmbeddings
    embeddings = OpenAIEmbeddings()    
    pdf_reader = PdfReader(resume)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
    resume_splitted = text_splitter.split_text(text)
    
    chroma_resume = Chroma.from_texts(
        resume_splitted,embeddings,
        collection_metadata={"hnsw:space": "cosine"} # l2 is the default (persist_directory="vector_storage/resume_store")
    )
    retriever_resume = chroma_resume.as_retriever(search_type = "similarity", search_kwargs = {"k":2})

    # Cache the embeddings and timestamp
    st.session_state.resumeEmbeddings = retriever_resume
    st.session_state.resumeTimestamp = time.time()
    return retriever_resume


def initialize_session_state_resume(input_text,resume):

     # Check if resume and job description are not empty
    if not resume or not input_text:
        raise ValueError("Resume and job description text must not be empty.")
    
    if "jdRetriever" not in st.session_state:
        st.session_state.jdRetriever = jd_retrieval(input_text)
    # convert resume to embeddings
    if "resumeRetriever" not in st.session_state:
        st.session_state.resumeRetriever = resume_retrieval(resume)
    if 'merger' not in st.session_state:
        st.session_state.merger = MergerRetriever(retrievers=[st.session_state.jdRetriever,st.session_state.resumeRetriever]) 
    if "job_chain_type_kwargs" not in st.session_state:
        interview_prompt = PromptTemplate(input_variables=["context","question"],
                                          template=Template.jd_template)
        st.session_state.job_chain_type_kwargs = {"prompt": interview_prompt}
    if "resume_history" not in st.session_state:
        st.session_state.resume_history = []
        st.session_state.resume_history.append(Message(origin="ai", message="Hello, I am your interivewer today. I will ask you some questions regarding your resume and your experience. Please start by saying hello or introducing yourself. Note: The maximum length of your answer is 4097 tokens!"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    # memory buffer for resume screen
    if "resume_memory" not in st.session_state:
        st.session_state.resume_memory = ConversationBufferMemory(human_prefix = "Candidate: ", ai_prefix = "Interviewer")
    #guideline for resume screen
    if "resume_guideline" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.5,)
        st.session_state.resume_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.job_chain_type_kwargs,
            chain_type='stuff',
            retriever=st.session_state.merger, 
            memory = st.session_state.resume_memory).run("Create an interview guideline and prepare only two questions for each topic. Make sure the questions tests the knowledge")
    # llm chain for resume screen
    if "resume_screen" not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7, )
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template= """I want you to act as an interviewer strictly following the guideline in the current conversation.
            
            Ask me questions and wait for my answers like a human. Do not write explanations.
            Candidate has no assess to the guideline.
            Only ask one question at a time. 
            Do ask follow-up questions if you think it's necessary.
            Do not ask the same question.
            Do not repeat the question.
            Candidate has no assess to the guideline.
            You name is Intelligent HR.
            I want you to only reply as an interviewer.
            Do not write all the conversation at once.
            Candiate has no assess to the guideline.
            
            Current Conversation:
            {history}
            
            Candidate: {input}
            AI: """)
        st.session_state.resume_screen = ConversationChain(prompt=PROMPT, llm = llm, memory = st.session_state.resume_memory)
    if "feedback" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.5,)
        st.session_state.feedback = ConversationChain(
            prompt=PromptTemplate(input_variables = ["history", "input"], template = Template.feedback_template),
            llm=llm,
            memory = st.session_state.resume_memory,
        )
# function to define the feedback of the interview
def show_feedback():
    if "feedback" in st.session_state:
        feedback_response = st.session_state.feedback.run(
            "please give evalution regarding the interview"
        )
        with st.expander("Evaluation"):
            st.write(feedback_response)

# st.sidebar.success("Select a demo above.")
st.set_page_config(page_title="AI-hr")
st.title("Interview AI")
def main():
    
    job_desc = st.text_area("Provide a brief description here", value=st.session_state.input_text if "input_text" in st.session_state else "")
    if job_desc:
        st.session_state.input_text = job_desc
        st.success("Job description saved.")
    
    resume = st.file_uploader("Upload your resume", type=["pdf"])
    if resume is not None:
        st.session_state.resume = resume
        st.success("Resume uploaded successfully.")

    # Submit button to start the interview simulation
    button = st.button("Submit")
    try:
        if button or "resume" in st.session_state and "input_text" in st.session_state:
            initialize_session_state_resume(st.session_state.input_text,st.session_state.resume)
            for message in st.session_state.resume_history:
                    with st.chat_message(message.origin):
                        st.markdown(message.message)  
            
            agree = st.checkbox("Access the Voice Assistant")
            if agree:
                #record audio input
                audio_bytes = audio_recorder(
                    pause_threshold=2.0, sample_rate=41_000,
                    text="",
                    recording_color="#fffff",
                    neutral_color="#6aa36f",
                    icon_name="microphone-lines",
                    icon_size="3x",
                    )
                if audio_bytes:
                    audio_input = "chatbot/audiofile.wav"
                    with open(audio_input,"wb") as f:
                        f.write(audio_bytes)

                    #convert speech to text
                    user_input = speech_to_text(audio_input)
                    with st.chat_message("human"):
                        st.markdown(user_input)
                        st.session_state.resume_history.append(Message(origin="human", message=user_input))
                        st.session_state.token_count += len(user_input.split())
                    st.audio(audio_bytes, format="audio/wav")
                    # generate bot response
                    bot_response = st.session_state.resume_screen.run(user_input)
                    with st.chat_message("assistant"):
                        st.markdown(f"Bot: {bot_response}")
                        st.session_state.resume_history.append(Message(origin="ai", message=bot_response))
                    # convert text to speech
                    audio_output = "chatbot/audiofileout.wav"
                    text_to_speech(speech_file_path=audio_output,chat_format=bot_response)
                    st.audio(audio_output) 
            else:
            
                # for message in st.session_state.resume_history:
                #     with st.chat_message(message.origin):
                #         st.markdown(message.message)

                if user_input := st.chat_input("Chat with me!"):
                    # Display user message in chat message container
                    with st.chat_message("human"):
                        st.markdown(user_input)
                    # Add user message to chat history
                    st.session_state.resume_history.append(Message(origin="human", message=user_input))
                    st.session_state.token_count += len(user_input.split())

                    # Generate bot response
                    bot_response = st.session_state.resume_screen.run(user_input)

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(f"Bot: {bot_response}")
                    
                    # Add assistant response to chat history
                    st.session_state.resume_history.append(Message(origin="ai", message=bot_response))
    

        else:
            st.warning("Please upload your resume and provide a job description before submitting.")
    except AttributeError:
        # raise "Please upload before "
        pass
    if st.button("Show feedback"):
        show_feedback()
# call the function
if __name__ == "__main__":
    main()

