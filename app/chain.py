import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Promtior, \
an AI consulting company. Use the following context retrieved from Promtior's website \
and documentation to answer the user's question accurately.

If the context does not contain enough information to answer the question, say: \
"I don't have information about that."

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    get_question = RunnableLambda(lambda x: x["question"])

    chain = (
        {
            "context": get_question | retriever | format_docs,
            "question": get_question,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
