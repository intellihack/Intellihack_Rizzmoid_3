from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import dotenv

dotenv.load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")


chat_history = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{inputt}"),
    ]
)

chain = prompt | chat

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="inputt",
    history_messages_key="chat_history",
)


loader = PyPDFLoader("loan_description.pdf")
pages = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(pages)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(k=4)


while True:
    inputt = input("Enter your question: ")
    # input = "I want to get a loan"
    response = chain_with_history.invoke(
        {"inputt": inputt, "context": retriever.invoke(inputt)},
        {"configurable": {"session_id": "unused"}},
    )
    print("\n", response.content, "\n")
