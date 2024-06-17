import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import FlatReader

from llama_index.llms.ollama import Ollama

st.header("Chat with the BattINFO ontology")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the classes in the BattINFO ontology."}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing BattINFO classes."):
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)
        Settings.embed_model = embed_model # we specify the embedding model to be used
        parser = FlatReader()
        file_extractor = {".txt": parser}
        documents = SimpleDirectoryReader(
            "./data", file_extractor=file_extractor
        ).load_data()
        vector_index = VectorStoreIndex.from_documents(documents)
        vector_index.storage_context.persist(persist_dir="./vector_storage")
        return vector_index
    
vector_index = load_data()

llm = Ollama(model="llama3", request_timeout=120.0)
Settings.llm = llm

query_engine = vector_index.as_query_engine(streaming=False, similarity_top_k=4)

qa_prompt_tmpl_str = (
            "Context information about the ontology is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"

            "Given the context information above I want you to think step by step to answer the query in a crisp manner, "
            "incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str} Output only a list of tags, seperated by a '^': ^CycleLife ^LithiumAirBattery ^R2012."
            "Make sure all tags are in the context information above.\n"
            "Answer: "
            )

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})



if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)