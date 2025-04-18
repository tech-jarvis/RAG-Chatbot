import re
from langchain.document_loaders import DirectoryLoader, CSVLoader
from pathlib import Path
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings



import re
import base64
import glob
import os
import textwrap
import time
from urllib.parse import urljoin

# import chromadb
import pinecone
import requests
import torch
from bs4 import BeautifulSoup

#from constants import CHROMA_SETTINGS
# from chromadb.config import Settings
from langchain import HuggingFaceHub, LLMChain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import (
    DirectoryLoader,
    PDFMinerLoader,
    PyPDFLoader,
    UnstructuredURLLoader,
)
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import load_prompt
#from project import settings


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    revision="gptq-4bit-32g-actorder_True",
    model_basename = model_basename,
    use_safetensors= True,
    trust_remote_code= True,
    inject_fused_attention=False,
    quantize_config=None,
    device= DEVICE,)
generation_config = GenerationConfig.from_pretrained(model_name_or_path)



llm = OpenAI(
    temperature = 0.1,
    model_name = "gpt-3.5-turbo"
)

embeddings = OpenAIEmbeddings()


def extract_urls_from_url(url):
    visited_urls = set()
    try:
        visited_urls.add(url)
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a", href=True)
            urls = []
            if links is not None:
                for link in links:
                    absolute_url = urljoin(url, link["href"])
                    if absolute_url not in visited_urls:
                        urls.append(absolute_url)
            return urls
        else:
            return []
    except requests.exceptions.InvalidSchema:
        print("InvalidSchema: No connection adapters were found.")



def Pinecone_Upsert(file=None, url=None,text=None):
    documents = []
    if file:
        # file = str(settings.BASE_DIR) + file
        # print(file)
        loader = PDFMinerLoader(file)
        documents.extend(loader.load())
    if url:
        urls = extract_urls_from_url(url)
        urls = list(set(urls)
        )
        loader_url = UnstructuredURLLoader(urls=urls)
        documents.extend(loader_url.load())
    if len(documents) > 0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=180)
        texts = text_splitter.split_documents(documents)
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        docsearch.add_documents(texts)
    if text:
        db = Pinecone(index, embeddings.embed_query, "text")
        db.add_texts(text)


def extract_urls_from_url(url):
    visited_urls = set()
    try:
        visited_urls.add(url)
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a", href=True)
            urls = []
            if links is not None:
                for link in links:
                    absolute_url = urljoin(url, link["href"])
                    if absolute_url not in visited_urls:
                        urls.append(absolute_url)
            return urls
        else:
            return []
    except requests.exceptions.InvalidSchema:
        print("InvalidSchema: No connection adapters were found.")



def Pinecone_Upsert(file=None, url=None,text=None):
    documents = []
    if file:
        # file = str(settings.BASE_DIR) + file
        # print(file)
        loader = PDFMinerLoader(file)
        documents.extend(loader.load())
    if url:
        urls = extract_urls_from_url(url)
        urls = list(set(urls)
        )
        loader_url = UnstructuredURLLoader(urls=urls)
        documents.extend(loader_url.load())
    if len(documents) > 0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=180)
        texts = text_splitter.split_documents(documents)
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        docsearch.add_documents(texts)
    if text:
        db = Pinecone(index, embeddings.embed_query, "text")
        db.add_texts(text)


# print(prompt)
def Pinecone_Upsert(file=None, url=None,csv=None, text=None):
    documents = []
    if file:
        # file = str(settings.BASE_DIR) + file
        file = os.path.join("/content", file)
        print(file)
        loader = PDFMinerLoader(file)
        documents.extend(loader.load())


    if url:
        urls = extract_urls_from_url(url)
        urls = list(set(urls))
        print(urls)
        loader_url = UnstructuredURLLoader(urls=urls)
        documents.extend(loader_url.load())
    if csv:
      csv = os.path.join("/content", csv)
      print(csv)
      loader = CSVLoader(csv, encoding='ISO-8859-1')
      documents.extend(loader.load())

    if len(documents) > 0:
        print("The loaded document: ",documents)
#NEW ADDITION [start]
        for _index in range(len(documents)):
            # Step 1: Access the page_content attribute of the custom object
            document = documents[_index]
            page_content = document.page_content
            #print(page_content)

            # Step 2: Perform transformations on the extracted text
            transformed_content = clean_sentence(page_content)

            #print(transformed_content)

            # Step 3: Update the page_content attribute with the transformed text
            document.page_content = transformed_content
            #print(document)

            # Step 4: Reconstruct the original structure with square brackets
            documents[_index] = document
            print("Transformed Document Final: ", documents[_index])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                       chunk_overlap=180
                                                       )
        # print("Splitter Address in meory: ", text_splitter)
#NEW ADDITION[end]


        texts = text_splitter.split_documents(documents)
        print("Splitted document: ",texts)
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        docsearch.add_documents(texts)



# # print(tokenizer.decode(output[0]))
pdf_file_path = "/content/drive/MyDrive/Colab Notebooks/"
Pinecone_Upsert(file=pdf_file_path)
# # #Call the Pinecone_Upsert function with the URL and file path:

# # # print(tokenizer.decode(output[0]))
# pdf_file_path = "/content/Updated_Services and Fees of Calliopée Business Center.pdf"
# #pdf_file_path = "https://www.calliopee.ch/"


# # # #Call the Pinecone_Upsert function with the URL and file path:

# Pinecone_Upsert(file=pdf_file_path)
#To Delete all the data
#index.delete(delete_all=True)

import os
# Define the folder path containing your PDF files
pdf_folder_path = "/content/drive/MyDrive/folder_pdfs"

# List all files in the folder
pdf_files = os.listdir(pdf_folder_path)

# Iterate through the PDF files and upsert them
for pdf_file in pdf_files:
    # Construct the full path to the PDF file
    pdf_file_path = os.path.join(pdf_folder_path, pdf_file)

    # Call the Pinecone_Upsert function with the PDF file path
    Pinecone_Upsert(file=pdf_file_path)

    # Optionally, you can print a message to indicate which PDF is being processed
    print(f"Upserted PDF: {pdf_file}")


streamer = TextStreamer(
    tokenizer, skip_prompt = True, skip_special_tokens= True, use_multiprocessing = False
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer= tokenizer,
    #max_length=4000,
    max_new_tokens = 1000,
    temperature=0,
    top_p= 0.95,
    repetition_penalty= 1.15,
    generation_config=generation_config,
    streamer = streamer,
    batch_size = 1,
)

llm= HuggingFacePipeline(pipeline=pipe)


# embeddings = HuggingFaceEmbeddings(
#     model_name= "embaas/sentence-transformers-multilingual-e5-base",
#     model_kwargs= {"device":DEVICE},
# )
embeddings = HuggingFaceInstructEmbeddings(
    model_name = "hkunlp/instructor-xl", model_kwargs={"device":DEVICE}
)

# embeddings = SentenceTransformerEmbeddings(
#     model_name="all-mpnet-base-v2",
#     encode_kwargs=dict(normalize_embeddings=True),
# )


text_field = "text"
db = Pinecone(index, embeddings.embed_query, text_field)


retriever = db.as_retriever(search_type="mmr",search_kwargs=dict(k= 4,
                                        fetch_k= 80,
                                        lambda_mult= 0.7,
))

# db.similarity_search(
#     query,  # our search query
#     k=3  # return 3 most relevant docs
# )
#NEW ADDITION[end]
#retriever = db.as_retriever(search_type="similarity",search_kwargs=dict(k= 2))


query = "Minimun cost of Domiciliation"
#db.similarity_search(query, k=3)
retriever.get_relevant_documents(query)


DEFAULT_TEMPLATE = """
### Instruction: You're an Virtual Assistant. Use only the chat history and the following information
{context} to answer the question. If you don't know the answer - say that you don't know.
Always reply to greetings in short and concise manner.
Keep your replies short, compassionate, and informative.
{chat_history}
### Input: {question}
### Response:
"""
prompt = PromptTemplate (
        input_variables = ["context", "question", "chat_history"],
        template = DEFAULT_TEMPLATE,
    )
memory = ConversationBufferMemory(
        memory_key = "chat_history",
        human_prefix = "### Input",
        ai_prefix = "### Response",
        input_key = "question",
        output_key = "output_text",
        return_messages = False,
    )


chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=prompt,
        memory=memory,
        verbose=False,
    )
db = Pinecone(index, embeddings.embed_query, "text")



while True:
  retriever = db.as_retriever(search_type="mmr")
  user_input=input("Enter: ")
  docs = retriever.get_relevant_documents(user_input)
  response = chain.run({"input_documents":docs, "question": user_input})
  print(response)


DEFAULT_TEMPLATE = """
Instruction: You're an Virtual Assistant. Use only the chat history and the following information
{context} to answer the question. If you don't know the answer - say that you don't know.
Always reply to greetings in short and concise manner.
Keep your replies short, compassionate, and informative.
{chat_history}
Input: {question}
Response: """


class Chatbot:
  def __init__(
      self,
      text_pipeline: llm,
      embeddings:embeddings,
      documents_dir:Path,
      prompt_template: str = DEFAULT_TEMPLATE,
      verbose: bool = False,
  ):
    prompt = PromptTemplate (
        input_variables = ["context", "question", "chat_history"],
        template = prompt_template,
    )
    self.chain = self._create_chain(text_pipeline, prompt, verbose)
    self.db = Pinecone(index, embeddings.embed_query, "text")
    self.retriever = db.as_retriever(search_type="mmr")

  def _create_chain(self,text_pipeline: llm, prompt: PromptTemplate,verbose: bool = False,):
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        human_prefix = "### Input",
        ai_prefix = "### Response",
        input_key = "question",
        output_key = "output_text",
        return_messages = False,
    )
    return load_qa_chain(
        text_pipeline,
        chain_type="stuff",
        prompt=prompt,
        memory=memory,
        verbose=False,

    )
  # def _embed_data(
  #     self, documents_dir: Path, embeddings: HuggingFaceEmbeddings
  # ) ->Pinecone:
  #     loader = DirectoryLoader(documents_dir, glob="**/*pdf")
  #     documents = loader.load()
  #     text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
  #     texts= text_splitter.split_documents(documents)
  #     return Pinecone.from_documents(texts, embeddings)
  def __call__(self,user_input:str)->str:
    #docs = self.db.similarity_search(user_input)
      docs = self.retriever.get_relevant_documents(user_input)
      answer = self.chain.run({"input_documents":docs, "question": user_input})
      return answer


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
while True:
  user_input = input("You: ")
  if user_input.lower() in ["bye", "goodbye", "quit", "exit"]:
    print(" Bye! It was nice chatting with you.")
    break
  answer = chatbot(user_input)
  print("\n")


  chain = load_qa_chain(llm,
                      chain_type="map_rerank",
                      return_intermediate_steps=True
                      )

query = "What are the rates of Domiciliations?"
docs = docsearch.similarity_search(query,k=10)
results = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
results