import json
from typing import Dict
import logging
import yaml
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddings:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def embed_query(self, query: str) -> list:
        return self.model.encode(query).tolist()

class DataLoader:
    def __init__(self, file_path: str = "data/horoscope.json"):
        self.file_path = file_path

    def load_data(self) -> Dict[str, str]:
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file)
        return data

class VectorStore:
    def __init__(self, data: Dict[str, str], embedding_function: LocalEmbeddings):
        self.data = data
        self.embedding_function = embedding_function
        self.chunk_size = 200
        self.chunk_overlap = 30
        self.text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.db_dict = self.create_vector_db()

    def create_vector_db(self) -> Dict[str, Chroma]:
        chunked_data_dict = {key: self.text_splitter.split_text(value) for key, value in self.data.items()}
        db_dict = {key: Chroma.from_texts(chunked_data_dict[key], self.embedding_function) for key in chunked_data_dict.keys()}
        return db_dict

class QuestionAnsweringSystem:
    def __init__(self, llm, tokenizer, star_sign: str, db_dict: Dict[str, Chroma], temperature: float = 0.6, k: int = 2):
        self.llm = llm
        self.tokenizer = tokenizer
        self.star_sign = star_sign
        self.db_dict = db_dict
        self.temperature = temperature
        self.k = k
        self.embeddings = LocalEmbeddings()
        self.vectorStore = self.db_dict[self.star_sign]
        self.retriever = self.vectorStore.as_retriever(search_kwargs={"k": self.k})
        self.initialize_chain()
        self.chat_history = []

    def initialize_chain(self):
        QUESTION_PROMPT = PromptTemplate.from_template("""
            Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
            This is a conversation with a human. Answer the questions you get based on the context.
            If you don't know the answer, just say that you don't, don't try to make up an answer. If the question is not relevant to the context,  you should steer the conversation towards the context.
            Chat History: {chat_history}
            Follow Up Input: {question}
            Standalone question:
            """)
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=QUESTION_PROMPT,
            return_source_documents=False,
            verbose=False
        )

    def conversational_chat(self, query: str) -> str:
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        return result["answer"]

def load_llm_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

    return llm, tokenizer

def start_chat(model_name):
    llm, tokenizer = load_llm_tokenizer(model_name)

    data_loader = DataLoader()
    data = data_loader.load_data()

    embeddings = LocalEmbeddings()
    vector_store = VectorStore(data, embeddings)
    db_dict = vector_store.create_vector_db()

    # List of star signs
    star_signs = list(db_dict.keys())

    # Let the user select the star sign
    selected_sign = st.sidebar.selectbox("Select your star sign", star_signs)

    qa_system = QuestionAnsweringSystem(llm, tokenizer, selected_sign, db_dict)

    st.title("Horoscope Chat")

    chat_history = []

    def chat(message):
        response = qa_system.conversational_chat(message)
        chat_history.append({"user": message, "assistant": response})
        return chat_history

    user_input = st.text_input("Enter your message:", key="input")
    if user_input:
        chat_history = chat(user_input)

    if chat_history:
        for chat in chat_history:
            st.markdown(f"**User:** {chat['user']}")
            st.markdown(f"**Assistant:** {chat['assistant']}")

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_name = config["model_id"]
    start_chat(model_name)
