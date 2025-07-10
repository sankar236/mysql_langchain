import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

identity = RunnableLambda(lambda x: x)

template = """
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)

prompt.format(schema="my schema", question="how many users are there?")

db_uri = "mysql+mysqlconnector://root:password@localhost:3306/chinook"
db = SQLDatabase.from_uri(db_uri)
res = db.run("SELECT * FROM Album LIMIT 5")
print(res)

def get_schema(_):
    return db.get_table_info()

llm = ChatOpenAI()

sql_chain = (
    identity.assign(schema=get_schema)
    | prompt
    | llm.bind(stop="\nSQL Result:")
    | StrOutputParser
)

res = sql_chain.invoke({"question":"how many artists are there?"})
print(res) ##print sql equivalant query to user entered prompt

template = """
Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
"""

prompt = ChatPromptTemplate.from_template(template)

def run_query(query):
    return db.run(query)

full_chain = (
    identity.assign(query=sql_chain).assign(
        schema = get_schema,
        response = lambda vars: run_query(vars["query"])
    )
    | prompt
    | llm
)

