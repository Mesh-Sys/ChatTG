# v0.6.1
VERSION = "v0.6.1"

import requests
import json
import re
import uuid
import threading
import textwrap
import logging
import datetime
import os
import subprocess

# define logging basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# if using sqlite
import sqlite3

#if using postgresql
# import psycopg2

# if using mariadb
# import mysql.connector

# add pyTelegramBotApi 
import telebot
from telebot import util

# default thread lock
thr_lock = threading.Lock()

# config location
WORKING_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIRECTORY = WORKING_DIRECTORY + "/config.json"

# CONSTANTS
# TELEGRAM CONSTANT VALUES
BOT_TOKEN = str()
CHAT_ID = str()

# DATABASE CONSTANT VALUE
CHAT_DATABASE_DIRECTORY = str()

# MODEL CONSTANT VALUES
OLLAMA_BASE_URL = str()
TEXT_MODEL = str() 

# LLM CONSTANTS
text_model_temperature = int()
text_model_num_thread = int()
text_model_num_ctx = int()
text_model_top_k = int()
text_model_top_p = int()

# GLOBAL ON EXECUTE VARIABLES
CURRENT_TEXT_CHAT_ID = str()

# START UTILITY FUNCTIONS
def log(msg):
    logging.info(msg)

def gen_uuid(prefix="a"):
    return (prefix + uuid.uuid4().hex)

def fs_mkdir(directory):
    try:
        os.mkdir(directory)
        log(f"CREATED {directory} SUCCESSFULLY")
    except OSError as e:
        log(f"ERROR OCCURED WHEN CREATING {directory} DIRECTORY, DIRECTORY PROBABLY EXISTS")
        log(f"{str(e)}\n")

def re_replace(text,new_text,input_text):
    return re.sub(text,new_text,input_text)

# END UTILITY FUNCTIONS

# START INIT FUNCTIONS
def load_config(config_directory):
    global BOT_TOKEN,CHAT_ID,CHAT_DATABASE_DIRECTORY,OLLAMA_BASE_URL,TEXT_MODEL
    global text_model_temperature,text_model_num_thread,text_model_num_ctx,text_model_top_k,text_model_top_p

    config_file = open(config_directory,"r")
    configs = json.load(config_file)
    config_file.close()

    BOT_TOKEN = configs["api_token"]
    CHAT_ID = configs["chat_id"]

    CHAT_DATABASE_DIRECTORY = configs["database_directory"]

    OLLAMA_BASE_URL = configs["ollama_api_url"]
    TEXT_MODEL = configs["text"]["user_set_model"]

    text_model_temperature = configs["text"]["model_settings"]["temperature"]
    text_model_num_thread = configs["text"]["model_settings"]["num_thread"]
    text_model_num_ctx = configs["text"]["model_settings"]["num_ctx"]
    text_model_top_k = configs["text"]["model_settings"]["top_k"]
    text_model_top_p = configs["text"]["model_settings"]["top_p"]

    log(f"BOT_TOKEN: {BOT_TOKEN}\n")
    log(f"CHAT_ID: {CHAT_ID}\n")
    log(f"CHAT_DATABASE_DIRECTORY: {CHAT_DATABASE_DIRECTORY}\n")
    log(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}\n")
    log(f"TEXT_MODEL: {TEXT_MODEL}\n")
    log(f"text_model_temperature: {text_model_temperature}\n")
    log(f"text_model_num_thread: {text_model_num_thread}\n")
    log(f"text_model_num_ctx: {text_model_num_ctx}\n")
    log(f"text_model_top_k: {text_model_top_k}\n")
    log(f"text_model_top_p: {text_model_top_p}\n")

def preload_config(config_directory):
    config_file = open(config_directory,"r")
    configs = json.load(config_file)
    config_file.close()
    return configs

def presave_config(config_directory, config_content):
    config_file = open(config_directory,"w")
    configs = json.dump(config_content, config_file, indent=4)
    config_file.close()

# END INIT FUNCTIONS

# START DATABASE FUNCTIONS
def sql_exec(database_file, sql, mode, values=None):
    database = sqlite3.connect(database_file)
    dbcursor = database.cursor()
    with thr_lock:
        if mode == 0:
            dbcursor.execute(sql)
            database.commit()
        elif mode == 1 and values != None:
            dbcursor.execute(sql,values)
            database.commit()
        elif mode == 2 and values != None:
            dbcursor.executemany(sql,values)
            database.commit()
    database.close()

def query_sql_exec(database_file, sql, mode, fetchmode, values=None):
    database = sqlite3.connect(database_file)
    dbcursor = database.cursor()
    result = None

    if mode == 0:
        dbcursor.execute(sql)
    elif mode == 1 and values != None:
        dbcursor.execute(sql,values)

    if fetchmode == 1:
        result = dbcursor.fetchone()
    elif fetchmode == 2:
        result = dbcursor.fetchall()

    database.close()

    if len(result) > 0:
        return result
    else:
        return None

def init_database(database_file):
    chat_table_sql = "CREATE TABLE IF NOT EXISTS chat_table (chat_id TEXT)"
    sql_exec(database_file, chat_table_sql, 0)

def get_all_chats_chat_id(database_file):
    chat_id_table_sql = "SELECT * FROM chat_table"
    return query_sql_exec(database_file, chat_id_table_sql, 0, 2)

def get_chats_with_chat_id(database_file, chat_id):
    query_uuid_chat_table_sql = "SELECT * FROM %s"
    query_uuid_chat_table_sql = re_replace("%s", chat_id, query_uuid_chat_table_sql)

    return query_sql_exec(database_file, query_uuid_chat_table_sql, 0, 2)

def get_single_row_chats_with_chat_id(database_file, chat_id):
    query_uuid_chat_table_sql = "SELECT * FROM %s"
    query_uuid_chat_table_sql = re_replace("%s", chat_id, query_uuid_chat_table_sql)

    return query_sql_exec(database_file, query_uuid_chat_table_sql, 0, 1)

def check_chat_id_on_chat_table(database_file, chat_id):
    check_chat_id_on_chat_table_sql = "SELECT * FROM chat_table WHERE chat_id = ?"
    values = (chat_id,)
    return query_sql_exec(database_file, check_chat_id_on_chat_table_sql, 1, 2, values)

def create_new_chat(database_file, chat_id):
    uuid_chat_table_sql = "CREATE TABLE IF NOT EXISTS %s (id TEXT, user TEXT, assistant TEXT)"
    uuid_chat_table_sql = re_replace("%s", chat_id, uuid_chat_table_sql)
    sql_exec(database_file, uuid_chat_table_sql, 0)

    chat_exists = check_chat_id_on_chat_table(database_file, chat_id)
    if chat_exists == None:
        insert_chat_table_sql = "INSERT INTO chat_table (chat_id) VALUES (?)"
        values = (chat_id,)
        sql_exec(database_file, insert_chat_table_sql, 1, values)

def insert_user_input(database_file, chat_id, input_data):
    unique_id = gen_uuid("cuid")
    insert_user_input_sql = "INSERT INTO %s (id, user, assistant) VALUES (?,?,?)"
    insert_user_input_sql = re_replace("%s", chat_id, insert_user_input_sql)
    values = (unique_id, input_data, "NULL")

    sql_exec(database_file, insert_user_input_sql, 1, values)
    return unique_id

def insert_assistant_input(database_file, chat_id, row_id, input_data):
    insert_assistant_input_sql = "UPDATE %s SET assistant = ? WHERE id = ?"
    insert_assistant_input_sql = re_replace("%s", chat_id, insert_assistant_input_sql)
    values = (input_data, row_id)

    sql_exec(database_file, insert_assistant_input_sql, 1, values)

    get_assistant_input_sql = "SELECT assistant FROM %s WHERE id = ?"
    get_assistant_input_sql = re_replace("%s", chat_id, get_assistant_input_sql)
    values = (row_id,)

    return query_sql_exec(database_file, get_assistant_input_sql, 1, 1, values)

# END DATABASE FUNCTIONS

# START LLM FUNCTIONS
# returns the generated string
def chat(model, message_list):
    url = OLLAMA_BASE_URL + "/api/chat"
    data = {
        "model" : model,
        "messages": message_list,
        "stream" : False,
        "options" : {
            "temperature" : text_model_temperature,
            "num_thread" : text_model_num_thread,
            "num_ctx" : text_model_num_ctx,
            "top_k" : text_model_top_k,
            "top_p" : text_model_top_p
        }
    }
    
    response = requests.post(url, json=data)

    if response.status_code == 200:
        chat_result = response.json()
        return chat_result["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# returns the generated string
def generate(model, prompt):
    url = OLLAMA_BASE_URL + "/api/generate"
    data = {
        "model" : model,
        "prompt" : prompt,
        "stream" : False,
        "options" : {
            "temperature" : text_model_temperature,
            "num_thread" : text_model_num_thread,
            "num_ctx" : text_model_num_ctx,
            "top_k" : text_model_top_k,
            "top_p" : text_model_top_p
        }
    }
    
    response = requests.post(url, json=data)

    if response.status_code == 200:
        generated_result = response.json()
        return generated_result["response"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# returns a list of models
def get_models():
    url = OLLAMA_BASE_URL + "/api/tags"

    response = requests.get(url)

    if response.status_code == 200:
        result = response.json()
        model_dict = result["models"]
        model_list = list()

        for model in model_dict:
            model_list.append(model["name"])
        
        return model_list
    else:
        return f"Error: {response.status_code} - {response.text}"

# returns a list of running models
def get_running_models():
    url = OLLAMA_BASE_URL + "/api/ps"

    response = requests.get(url)

    if response.status_code == 200:
        result = response.json()
        model_dict = result["models"]
        model_list = list()

        for model in model_dict:
            model_list.append(model["name"])
        
        return model_list
    else:
        return f"Error: {response.status_code} - {response.text}"

# returns a bool
def load_model(model):
    url = OLLAMA_BASE_URL + "/api/generate"
    data = {
        "model" : model,
    }
    
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["done"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# END LLM FUNCTIONS

# INITIALIZE INTERFACE
load_config(CONFIG_DIRECTORY)

fs_mkdir(f"{WORKING_DIRECTORY}/databases")

CURRENT_TEXT_CHAT_ID = gen_uuid("ch")

init_database(CHAT_DATABASE_DIRECTORY)

# START INTERFACING FUNCTIONS
def ai_chat(prompt,chat_id):
    create_new_chat(CHAT_DATABASE_DIRECTORY, CURRENT_TEXT_CHAT_ID)
    
    try:
        chats = get_chats_with_chat_id(CHAT_DATABASE_DIRECTORY, chat_id)
    except Exception as e:
        log(f"PRINT CHAT ERROR AI CHAT\n {str(e)}")
        return "EMPTY CHAT"

    if chats == None:
        log("EMPTY AI CHAT ATTEMPT INSERTION")

        unique_id = insert_user_input(CHAT_DATABASE_DIRECTORY, chat_id, prompt)
        message_data = [{
            "role": "user",
            "content": prompt
        }]
        response = insert_assistant_input(CHAT_DATABASE_DIRECTORY, chat_id, unique_id, chat(TEXT_MODEL, message_data))
        return response[0]
    else:
        chat_list = list()
        for db_prompt in chats:
            user_prompt = {
                "role": "user",
                "content": db_prompt[1]
            }
            chat_list.append(user_prompt)

            assistant_response = {
                "role": "assistant",
                "content": db_prompt[2]
            }
            chat_list.append(assistant_response)

        new_user_prompt = {
            "role": "user",
            "content": prompt
        }
        chat_list.append(new_user_prompt)
        unique_id = insert_user_input(CHAT_DATABASE_DIRECTORY, chat_id, prompt)
        response = insert_assistant_input(CHAT_DATABASE_DIRECTORY, chat_id, unique_id, chat(TEXT_MODEL, chat_list))
        return response[0]


def ai_generate_text(prompt):
    return generate(TEXT_MODEL,prompt)

def print_chat_with_chat_id(chat_id):
    try:
        chats = get_chats_with_chat_id(CHAT_DATABASE_DIRECTORY, chat_id)
    except Exception as e:
        log(f"PRINT CHAT ERROR\n {str(e)}")
        return "EMPTY CHAT"

    chat_string = str()

    if chats == None:
        log("EMPTY CHAT")
        return "EMPTY CHAT"
    else:
        for prompt in chats:
            chat_string = chat_string + f"\n`USER`:\n{prompt[1]}\n\n`AI`:\n{prompt[2]}\n"

    return chat_string

def get_chat_list_with_snippet():
    chats_chat_id = get_all_chats_chat_id(CHAT_DATABASE_DIRECTORY)

    chat_list_string = str()

    if chats_chat_id == None:
        log("NO CHATS AT ALL")
        return "NO CHATS AT ALL"
    else:
        for chat_id in chats_chat_id:
            try:
                chat_id_chats = get_single_row_chats_with_chat_id(CHAT_DATABASE_DIRECTORY, chat_id[0])

                if chat_id_chats == None:
                    log("EMPTY CHAT")
                    return "EMPTY CHAT"
                else:
                    chat_list_string = chat_list_string + f"`{chat_id[0]}`\n{chat_id_chats[1]}\n\n"

            except Exception as e:
                log(f"PRINT CHAT SNIPPET ERROR 1\n {str(e)}")

    return chat_list_string

def get_help_message():
    help_msg = textwrap.dedent(r"""
    `/start` - prints out the bot version
    `/help` or `/h` - prints out the help message
    `/ai` - uses the chat interface for chat completion
    `/chat` - uses the chat interface for chat completion
    `/generate` - uses the generate interface for generate completion
    `/newchat` or `/nc` - creates a new chat by generating a new uuid and adding a new database entry, returns chat id
    `/get_text_models` - returns a list of text llm models
    `/get_running_models` - returns a list of currently running models
    `/get_current_chat_id` - returns the current chat id
    `/load_text_model` - if no option is specified it loads the default model into memmory
    `/set_text_model` - set a text model
    `/get_chat_list` - show all chats with snippets
    `/get_current_text_model` - return the model to used for query
    `/print_chat` - prints a chat history
    `/select_chat` - select a chat for chating
    
    ------get commands
    /start
    /help, /h
    /newchat
    /get\_current\_chat\_id
    /get\_text\_models
    /get\_running\_models
    /get\_chat\_list
    /get\_current\_text\_model
    
    ------send commands
    `/ai` prompt
    `/chat` prompt
    `/generate` prompt
    `/load_text_model` model\_name
    `/set_text_model` model\_name
    `/print_chat` chat\_id
    `/select_chat` chat\_id
    """)
    return help_msg

def get_running_text_models():
    model_list = get_running_models()
    msg = "RUNNING MODELS\n"

    for model in model_list:
        msg = msg +  f"`{model}`\n"

    return msg

def get_text_models():
    model_list = get_models()
    msg = "AVAILABLE MODELS\n"

    for model in model_list:
        msg = msg +  f"`{model}`\n"

    return msg

# END INTERFACING FUNCTIONS

# START TELEGRAM BOT FUNCTIONS
# NOTE: ADD MARKDOWN FORMATING
bot = telebot.TeleBot(BOT_TOKEN)

def print_bot(message,chat_id,ps=None):
    split_message = util.smart_split(message, chars_per_string=3000)
    try:
        for msg in split_message:
            if ps == "mk":
                bot.send_message(chat_id,msg,parse_mode='Markdown')
            elif ps == "mk2":
                bot.send_message(chat_id,msg,parse_mode='MarkdownV2')
            else:
                bot.send_message(chat_id,msg)
    except Exception as e:
        log("AN ERROR OCCURED WHEN TRYING TO SEND MESSAGE THROUGH TELEGRAM API")
        log(str(e))

@bot.message_handler(commands=['start'])
def exec_cmd_start(message):
    msg = f"""
    ChatTG `{VERSION}`
    """
    print_bot(msg, CHAT_ID, "mk")

@bot.message_handler(commands=['help', 'h'])
def exec_cmd_help(message):
    print_bot(get_help_message(), CHAT_ID, "mk")

@bot.message_handler(commands=['newchat', 'nc'])
def exec_cmd_newchat(message):
    global CURRENT_TEXT_CHAT_ID
    CURRENT_TEXT_CHAT_ID = gen_uuid("ch")
    print_bot(f"NEW CHAT ID: `{CURRENT_TEXT_CHAT_ID}`", CHAT_ID, "mk")

@bot.message_handler(commands=['get_current_chat_id'])
def exec_cmd_get_current_chat_id(message):
    print_bot(f"CURRENT CHAT ID: `{CURRENT_TEXT_CHAT_ID}`", CHAT_ID, "mk")

@bot.message_handler(commands=['get_text_models'])
def exec_cmd_get_text_models(message):
    print_bot(get_text_models(),CHAT_ID,"mk")

@bot.message_handler(commands=['get_running_models'])
def exec_cmd_get_running_models(message):
    print_bot(get_running_text_models(),CHAT_ID,"mk")

@bot.message_handler(commands=['get_chat_list'])
def exec_cmd_get_chat_list(message):
    print_bot(get_chat_list_with_snippet(), CHAT_ID, "mk")

@bot.message_handler(commands=['get_current_text_model'])
def exec_cmd_get_current_text_model(message):
    print_bot(f"CURRENT TEXT MODEL: `{TEXT_MODEL}`",CHAT_ID, "mk")

@bot.message_handler(commands=['ai'])
def exec_cmd_ai(message):
    cmd_text = re_replace("/ai ","",message.text)
    print_bot("Running...",CHAT_ID,"mk")
    print_bot(ai_chat(cmd_text, CURRENT_TEXT_CHAT_ID), CHAT_ID, "mk")

@bot.message_handler(commands=['chat'])
def exec_cmd_chat(message):
    cmd_text = re_replace("/chat ","",message.text)
    print_bot("Running...",CHAT_ID,"mk")
    print_bot(ai_chat(cmd_text, CURRENT_TEXT_CHAT_ID), CHAT_ID, "mk")

@bot.message_handler(commands=['generate'])
def exec_cmd_generate(message):
    cmd_text = re_replace("/generate ","",message.text)
    print_bot("Running...",CHAT_ID,"mk")
    print_bot(ai_generate_text(cmd_text), CHAT_ID)

@bot.message_handler(commands=['load_text_model'])
def exec_cmd_load_text_model(message):
    cmd_text = re.split("\\s",message.text)

    configs = preload_config(CONFIG_DIRECTORY)
    configs["text"]["user_set_model"] = str(cmd_text[1])
    presave_config(CONFIG_DIRECTORY,configs)
    load_config(CONFIG_DIRECTORY)

    print_bot("Loading...",CHAT_ID,"mk")
    load_model(TEXT_MODEL)
    print_bot(get_running_text_models(),CHAT_ID,"mk")

@bot.message_handler(commands=['set_text_model'])
def exec_cmd_set_text_model(message):
    cmd_text = re.split("\\s",message.text)

    configs = preload_config(CONFIG_DIRECTORY)
    configs["text"]["user_set_model"] = str(cmd_text[1])
    presave_config(CONFIG_DIRECTORY,configs)
    load_config(CONFIG_DIRECTORY)

    print_bot(f"CURRENT TEXT MODEL: `{TEXT_MODEL}`", CHAT_ID, "mk")

@bot.message_handler(commands=['print_chat'])
def exec_cmd_print_chat(message):
    cmd_text = re.split("\\s", message.text)

    print_bot(print_chat_with_chat_id(str(cmd_text[1])), CHAT_ID, "mk")

@bot.message_handler(commands=['select_chat'])
def exec_cmd_select_chat(message):
    global CURRENT_TEXT_CHAT_ID

    cmd_text = re.split("\\s", message.text)

    CURRENT_TEXT_CHAT_ID = str(cmd_text[1])
    print_bot(f"CURRENT CHAT ID: `{CURRENT_TEXT_CHAT_ID}`", CHAT_ID, "mk")

bot.infinity_polling()
# END TELEGRAM BOT FUNCTIONS