import os
import json
import csv
import psycopg2
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


client = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://litellm.aks-hs-prod.int.hyperskill.org",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

# Initialize embeddings model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url="https://litellm.aks-hs-prod.int.hyperskill.org",
    api_key=os.getenv("OPENAI_API_KEY"),
)


conn = psycopg2.connect("postgresql://hyper:hyper2025@localhost:5432/hyperdb")
cursor = conn.cursor()

cursor.execute("""
    DROP TABLE IF EXISTS orders;
""")
conn.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        orders_date TEXT,
        orders_id TEXT PRIMARY KEY,
        customer_id TEXT,
        item TEXT,
        category TEXT,
        shipping_address TEXT,
        total_amount REAL
    )
""")
conn.commit()


with open('/home/ryzen/PycharmProjects/Advanced RAG1/Advanced RAG/task/sales_data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    sample_orders = list(reader)

insert_query = """
    INSERT INTO orders (orders_date, orders_id, customer_id, item, category, shipping_address, total_amount)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (orders_id) DO NOTHING
"""

cursor.executemany(insert_query, sample_orders)
conn.commit()
cursor.close()


knowledge_base_content = {
    "knowledge-base": [
        {"question": "How long does shipping take?",
         "answer": "Our shipping policy ensures delivery within 5-7 business days."},
        {"policy": "Our privacy policy outlines how we handle your data."},
        {"policy": "We do not have a specific return policy detailed here. For returns, please contact support directly."},
        {"how-to": ["To reset your password, go to the login page and click 'Forgot Password'."]},
        {"support": "You can contact our customer support team on our website at https://support.hypersite.org or by email at hello@hypersite.org."},
        {"support": "Our team is available 24/7 to assist you with any inquiries or issues you may have."},
        {"support": "We offer live chat support on our website during our business hours."}
    ]
}

knowledge_base_filename = "knowledge_base_noisy.json"
if not os.path.exists(knowledge_base_filename):
    with open(knowledge_base_filename, "w") as f:
        json.dump(knowledge_base_content, f, indent=4)

with open(knowledge_base_filename, "r") as f:
    raw_data = json.load(f)

all_documents = []
for entry in raw_data.get("knowledge-base", []):
    if "question" in entry and "answer" in entry:
        text_block = f"question: {entry['question']}\nanswer: {entry['answer']}"
        all_documents.append(text_block)
    elif "policy" in entry:
        all_documents.append(f"policy: {entry['policy']}")
    elif "how-to" in entry:
        all_documents.extend([f"how-to: {step}" for step in entry["how-to"]])
    elif "support" in entry:
        all_documents.append(f"support: {entry['support']}")


vectorstore = FAISS.from_texts(all_documents, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for info on policies, FAQs, how-tos, and support.
    """
    print(f"Searching knowledge base for: '{query}'")
    docs = retriever.invoke(query)  # <-- updated here
    if not docs:
        return "No relevant information found in the knowledge base."

    context = "\n".join([doc.page_content for doc in docs])
    print(f"Relevant chunks: {context}")
    return context


@tool
def get_order_details_from_db(order_id: str) -> str:
    """
    Fetch order details from the 'orders' table by order_id.
    Returns a list of tuples with raw values.
    """
    print(f"Fetching details for order_id: '{order_id}'")
    try:
        with psycopg2.connect("postgresql://hyper:hyper2025@localhost:5432/hyperdb") as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT * FROM orders WHERE orders_id = %s', (order_id,))
                result = cursor.fetchall()
                if not result:
                    return f"No order found with ID '{order_id}'."
                print(f"Order details: {result}")
                return str(result)
    except Exception as e:
        print(f"DB error: {e}")
        return "An error occurred while fetching order details."



tools = [search_knowledge_base, get_order_details_from_db]

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful customer support assistant. You must use the provided tools to find information to answer the user's questions. "
     "If the user asks about returns, state that you don't have specific information and advise them to contact support. "
     "Combine information from multiple tools if necessary to provide a comprehensive response."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


agent = create_openai_tools_agent(client, tools, prompt)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False  # Set True to see reasoning steps
)

if __name__ == "__main__":
    chat_history = []
    while True:
        try:
            user_query = input("> ").strip()
            if user_query.lower() in ['exit', 'quit']:
                break

            result = agent_executor.invoke({
                "input": user_query,
                "chat_history": chat_history
            })
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=result["output"]))

            if len(chat_history) > 12:
                chat_history = chat_history[-12:]

            print(result["output"])

        except Exception as e:
            print(f"An error occurred: {e}")
conn.close()
