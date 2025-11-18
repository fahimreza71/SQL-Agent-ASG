import os
from dotenv import load_dotenv
from sqlalchemy.engine.url import URL
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
import logging
import urllib
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Load Config and Create Agent (Global) ---
logger.info("Starting server, loading environment variables...")
load_dotenv()

# DB credentials
server = os.getenv("MS_SQL_SERVER")
database = os.getenv("MS_SQL_DATABASE")
username = os.getenv("MS_SQL_USER")
password = os.getenv("MS_SQL_PASSWORD")
driver = os.getenv("MS_SQL_DRIVER", "ODBC Driver 17 for SQL Server")

google_api_key = os.environ["GOOGLE_API_KEY"]
model_name = os.getenv("gpt_deployment_name")

if not all([server, database, google_api_key]):
    logger.error("CRITICAL: Missing required environment variables. Check .env file.")
    raise ValueError("Error: Missing required environment variables. Check .env file.")

password_enc = urllib.parse.quote_plus(password)
driver_enc = urllib.parse.quote_plus(driver)

db_uri = f"mssql+pyodbc://@{server}/{database}?driver={driver_enc}"
# include_tables = ['Employee', 'Office', 'OfficeTeam', 'OfficeTeamEmployee']

try:
    # db = SQLDatabase.from_uri(str(db_uri), include_tables=include_tables)
    db = SQLDatabase.from_uri(str(db_uri))
    logger.info(f"Connected to DB. Using tables: {db.get_table_names()}")
except Exception as e:
    logger.critical(f"CRITICAL Error connecting to database: {e}")
    exit(1)

# Load LLM
logger.info("Loading Google Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=google_api_key,
    temperature=0.0
)

system_message = """
You are a helpful SQL agent. Your name is DataBot.
Do not answer questions about sensitive data like passwords.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

logger.info("Creating SQL Agent...")
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt=prompt_template,
    agent_type="openai-tools",
    verbose=True,
    handle_parsing_errors=True,
    # return_intermediate_steps=True
)

class GeminiSanitizedHistory(ChatMessageHistory):
    def add_user_message(self, message: str | BaseMessage):
        if isinstance(message, BaseMessage):
            content = message.content
        else:
            content = str(message)
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, message: str | BaseMessage):
        if isinstance(message, BaseMessage):
            content = message.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
        else:
            content = str(message)
        self.messages.append({"role": "assistant", "content": content})

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = GeminiSanitizedHistory()

    # messages_text = []
    # for msg in store[session_id].messages:
    #     if isinstance(msg, AIMessage):
    #         role = "ai"
    #         content = msg.content
    #     elif isinstance(msg, HumanMessage):
    #         role = "user"
    #         content = msg.content
    #     else:
    #         role = "unknown"
    #         content = str(msg)
    #     messages_text.append(f"{role}: {content}")

    # print("HISTORY:\n" + "\n".join(messages_text))

    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def run_agent(question, session_id):
    result = agent_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    if isinstance(result, list):
        result = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in result
        )

    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"])
        if "text" in result:
            return str(result["text"])

    return str(result)

logger.info("--- SQL Agent is Ready ---")

app = Flask(__name__)

@app.route("/ask", methods=['POST'])
def ask_agent():
    data = request.json

    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    question = data.get('question')
    session_id = data.get('session_id', 'default_user') 

    if not question:
        logger.warning("API call missing 'question' in JSON body")
        return jsonify({"error": "No question provided. Use {'question': '...'}"}), 400

    logger.info(f"Session: {session_id} | Question: {question}")

    history = get_session_history(session_id)
    history.add_user_message(question)

    try:
        response = run_agent(question, session_id)        

        clean_text = re.sub(r'[*]+', '', response).strip()

        # for item in response["output"]:
        #     if isinstance(item, dict) and "text" in item:
        #         full_text += item["text"]
        #     elif isinstance(item, str):
        #         full_text += item

        # clean_text = re.sub(r'[*]+', '', full_text)
        # clean_text = re.sub(r'\s*\n\s*', '\n', clean_text)
        # clean_text = clean_text.strip()
        history.add_ai_message(response)
        return jsonify({
            "answer": clean_text,
            # "intermediate_steps": str(response["intermediate_steps"])
        })
        
    except Exception as e:
        logger.error(f"Error during agent invocation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)