## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

## PROBLEM STATEMENT: 
Design an LCEL pipeline using LangChain with at least two dynamic prompt parameters.  Integrate prompt, model, and output parser components to form a complete expression.  Evaluate its functionality through real-world query-response scenarios.

## DESIGN STEPS:

#### STEP 1: Setup API and Environment: Load environment variables using dotenv and set openai.api_key from the local environment.

#### STEP 2: Create Prompt and Model: Use LangChain to define a ChatPromptTemplate and initialize ChatOpenAI for text generation.

#### STEP 3: Build a Retrieval System: Store predefined texts in DocArrayInMemorySearch with OpenAIEmbeddings and create a retriever.

#### STEP 4: Define Question-Answering Chain: Use RunnableMap to fetch relevant documents and pass them to a chat model for responses.

#### STEP 5: Invoke the Chain: Run chain.invoke() with a question to retrieve context-based answers using the LangChain pipeline.

## PROGRAM:
### Simple Chain
```python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "tell me about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser


response = chain.invoke({"topic": "England"})
print(response)
print('Name: AMIRTHA VARSHINI M')
print('Reg. No. : 212224230017')
```
### Complex Chain
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

model = ChatOpenAI()
output_parser = StrOutputParser()

vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "The Eiffel Tower is famous for being a historic landmark in Paris, France, known for its iron structure and popularity among tourists.",
 
        "People visit beaches for relaxation, swimming, vacations, and enjoying natural scenery.",

        "Exercise improves physical fitness, health, energy levels, and mental well-being.",

        "Books are important because they provide knowledge, improve imagination, and support learning."
    ],
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

print(retriever.get_relevant_documents("What is Python?"))
print(retriever.get_relevant_documents("What is Machine Learning?"))

template = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

print("Name : AMIRTHA VARSHINI M")
print("Reg. No. : 212224230017")

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

result1 = chain.invoke({
    "question": "What is the Eiffel Tower famous for?"
})

print(result1)

result2 = chain.invoke({
    "question": "Why do people visit beaches"
})

print(result2)

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

print(inputs.invoke({
    "question": "What are the benefits of exercise?"
}))
```


## OUTPUT:
### Simple Chain

<img width="1330" height="424" alt="image" src="https://github.com/user-attachments/assets/f1e03d5c-cf45-4044-85e2-ebbdeae3f020" />

 ### Complex Chain

 <img width="1366" height="448" alt="image" src="https://github.com/user-attachments/assets/eebc1ee8-87be-4388-a727-70931e4ab5c1" />


### RESULT:

The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
