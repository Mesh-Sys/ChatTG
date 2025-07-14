# v0.6.2
VERSION = "v0.6.3"

import aiohttp
import asyncio
import uvloop
import json
import re
import uuid
import textwrap
import logging
import base64
from datetime import datetime
import os
from pydub import AudioSegment
from typing import Dict, Any, List

# ENABLE UVLOOP
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

LOG_FILE = "/var/log/chat_tg.log"
# define logging basic config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#logger_fh = logging.FileHandler(LOG_FILE, mode='w')
#logger_fh.setLevel(logging.INFO)
#logger_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger_sh = logging.StreamHandler()
logger_sh.setLevel(logging.INFO)
logger_sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
#logger.addHandler(logger_fh)
logger.addHandler(logger_sh)

# if using sqlite
import aiosqlite

# add pyTelegramBotApi 
import telebot
from telebot import util
from telebot.async_telebot import AsyncTeleBot

# config location
WORKING_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIRECTORY = f"{WORKING_DIRECTORY}/config.json"

# CONSTANTS
# TELEGRAM CONSTANT VALUES
BOT_TOKEN = str()
CHAT_ID = str()

# DATABASE CONSTANT VALUE
CHAT_DATABASE_DIRECTORY = str()

# MODEL CONSTANT VALUES
OLLAMA_BASE_URL = str()
TEXT_MODEL = str()
VOICE_MODEL = str()
WHISPER = dict()

# LLM CONSTANTS
text_model_temperature = int()
text_model_num_thread = int()
text_model_num_ctx = int()
text_model_top_k = int()
text_model_top_p = int()

# TOKENS
HF_TOKEN = str()

# GLOBAL ON EXECUTE VARIABLES
CURRENT_TEXT_CHAT_ID = str()

DATABASE = None

# INITIALIZE INTERFACE
CURRENT_TEXT_CHAT_ID = None

# INTIALIZE AIOHTTP
aiohttp_session = None

# START UTILITY FUNCTIONS
def gen_uuid(prefix: str = "a", full: bool = False) -> str:
	return f"{prefix}{uuid.uuid4().hex if full else str(uuid.uuid4()).split('-')[4]}"

async def _mkdir(directory: str | List[str]) -> None:
	async def create_directory(directory_path: str) -> None:
		try:
			if not os.path.exists(directory_path):
				os.mkdir(directory_path)
				logger.info(f"_mkdir - Created {directory_path} successfully")
		except Exception as e:
			logger.error(f"_mkdir - Error occured when creating {directory_path} directory - {e}\n")
	dir_state = [await create_directory(directory_entry) for directory_entry in directory] if type(directory) is list else await create_directory(directory)

def re_replace(text: str, new_text: str, input_text: str) -> str:
	return re.sub(text, new_text, input_text)

async def make_request(url: str, is_post: bool = False, data: Dict[str, Any] = None) -> Any:
	logger.info(f"make_request - Making new request - {url} - {data}")
	response = await aiohttp_session.post(url, json=data) if is_post else await aiohttp_session.get(url)
	return response.status, await response.text()

# END UTILITY FUNCTIONS

# START INIT FUNCTIONS
def load_config(config_directory: str) -> None:
	global BOT_TOKEN, CHAT_ID, CHAT_DATABASE_DIRECTORY, OLLAMA_BASE_URL, TEXT_MODEL, VOICE_MODEL, WHISPER, HF_TOKEN
	global text_model_temperature, text_model_num_thread, text_model_num_ctx, text_model_top_k, text_model_top_p

	configs = None
	with open(config_directory, "r") as config_file:
		configs = json.load(config_file)

	BOT_TOKEN = configs["api_token"]
	CHAT_ID = configs["chat_id"]

	CHAT_DATABASE_DIRECTORY = configs["database_directory"]

	OLLAMA_BASE_URL = configs["ollama_api_url"]
	TEXT_MODEL = configs["text"]["user_set_model"]
	VOICE_MODEL = configs["voice"]["user_set_model"]
	WHISPER = configs["whisper"]

	HF_TOKEN = configs["hf_token"]

	text_model_temperature = configs["text"]["model_settings"]["temperature"]
	text_model_num_thread = configs["text"]["model_settings"]["num_thread"]
	text_model_num_ctx = configs["text"]["model_settings"]["num_ctx"]
	text_model_top_k = configs["text"]["model_settings"]["top_k"]
	text_model_top_p = configs["text"]["model_settings"]["top_p"]

	logger.info("load_config - Loaded configs")
	logger.info(f"load_config - BOT_TOKEN: {BOT_TOKEN}")
	logger.info(f"load_config - CHAT_ID: {CHAT_ID}")
	logger.info(f"load_config - CHAT_DATABASE_DIRECTORY: {CHAT_DATABASE_DIRECTORY}")
	logger.info(f"load_config - OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
	logger.info(f"load_config - TEXT_MODEL: {TEXT_MODEL}")
	logger.info(f"load_config - VOICE_MODEL: {VOICE_MODEL}")
	logger.info(f"load_config - WHISPER: {WHISPER}")
	logger.info(f"load_config - HF_TOKEN: {HF_TOKEN}")
	logger.info(f"load_config - text_model_temperature: {text_model_temperature}")
	logger.info(f"load_config - text_model_num_thread: {text_model_num_thread}")
	logger.info(f"load_config - text_model_num_ctx: {text_model_num_ctx}")
	logger.info(f"load_config - text_model_top_k: {text_model_top_k}")
	logger.info(f"load_config - text_model_top_p: {text_model_top_p}")

# LOAD CONFIG
load_config(CONFIG_DIRECTORY)
logger.info("main - Loaded configs")

# INITIALIZE BOT
bot = AsyncTeleBot(BOT_TOKEN)

async def preload_config(config_directory: str) -> Dict[str, Any]:
	logger.info("preload_config - Preloading config")
	with open(config_directory, "r") as config_file:
		return json.load(config_file)

async def presave_config(config_directory: str, config_content: Dict[str, Any]) -> None:
	logger.info("presave_config - Presaving config")
	with open(config_directory, "w") as config_file:
		json.dump(config_content, config_file, indent=4)

# END INIT FUNCTIONS

# START DATABASE FUNCTIONS
async def sql_exec(sql: str, values: tuple = (), executemany: bool = False) -> Any:
	logger.info(f"sql_exec - Executing sql - {sql} - values - {values}")
	cursor = await DATABASE.executemany(sql, values) if executemany else await DATABASE.execute(sql, values)
	await DATABASE.commit()
	return cursor

async def query_sql_exec(sql: str, values: tuple = (), fetchall: bool = False) -> Any:
	logger.info(f"query_sql_exec - Querying sql - {sql} - values - {values}")
	cursor = await DATABASE.execute(sql, values)
	return await cursor.fetchall() if fetchall else await cursor.fetchone()

async def init_database(database_file: str) -> None:
	global DATABASE
	DATABASE = await aiosqlite.connect(database_file)
	DATABASE.row_factory = aiosqlite.Row
	await sql_exec("CREATE TABLE IF NOT EXISTS chats_list (created_at TEXT DEFAULT CURRENT_TIMESTAMP, chat_id TEXT NOT NULL UNIQUE)")
	await sql_exec("""CREATE TABLE IF NOT EXISTS chats (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		created_at TEXT DEFAULT CURRENT_TIMESTAMP,
		chat_id TEXT NOT NULL,
		type TEXT NOT NULL,
		user TEXT,
		assistant TEXT
	)""")

async def get_all_chats_chat_id() -> List[str]:
	logger.info(f"get_all_chats_chat_id - Fetching chats")
	chat_list = await query_sql_exec("SELECT * FROM chats_list", fetchall=True)
	return [chat_id["chat_id"] for chat in chat_list]

async def get_chats_with_chat_id_text(chat_id: str) -> Any:
	return await query_sql_exec("SELECT * FROM chats WHERE chat_id = ? AND type = 'text'", values=(chat_id,), fetchall=True)

async def get_single_row_chats_with_chat_id(chat_id: str) -> Any:
	return await query_sql_exec("SELECT 1 FROM chats WHERE chat_id = ? AND type = 'text'", values=(chat_id,))

async def create_new_chat(chat_id: str) -> None:
	try:
		await sql_exec("INSERT INTO chats_list (chat_id) VALUES (?)", values=(chat_id,))
	except aiosqlite.IntegrityError as e:
		logger.error(f"create_new_chat - Error creating new chat - {e}")

async def insert_user_input(chat_id: str, input_data: str) -> int:
	cursor = await sql_exec("INSERT INTO chats (chat_id, user) VALUES (?, ?) RETURNING id", values=(chat_id, input_data))
	return (await cursor.fetchone())["id"]

async def insert_assistant_input(row_id: int, input_data: str) -> Any:
	cursor = await sql_exec("UPDATE chats SET assistant = ? WHERE id = ? RETURNING assistant", values=(input_data, row_id))
	return (await cursor.fetchone())["assistant"]

# END DATABASE FUNCTIONS

# START LLM FUNCTIONS
# returns the generated string
async def chat(model: str, message_list: List[Any]) -> Any:
	status_code, response = None, None
	try:
		status_code, response = await make_request(
			url=f"{OLLAMA_BASE_URL}/api/chat",
			is_post=True,
			data={
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
		)
		return json.loads(response)["message"]["content"]
	except Exception as e:
		logger.error(f"chat - Error - {status_code} - {response} - {e}")

# returns the generated string
async def generate(model: str, prompt: str) -> Any:
	status_code, response = None, None
	try:
		status_code, response = await make_request(
			url=f"{OLLAMA_BASE_URL}/api/generate",
			is_post=True,
			data={
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
		)
		return json.loads(response)["response"]
	except Exception as e:
		logger.error(f"generate - Error - {status_code} - {response} - {e}")

# returns a list of models
async def get_models() -> Any:
	status_code, response = None, None
	try:
		status_code, response = await make_request(
			url=f"{OLLAMA_BASE_URL}/api/tags",
		)
		return [model["name"] for model in json.loads(response)["models"]]
	except Exception as e:
		logger.error(f"get_models - Error - {status_code} - {response} - {e}")

# returns a list of running models
async def get_running_models() -> Any:
	status_code, response = None, None
	try:
		status_code, response = await make_request(
			url=f"{OLLAMA_BASE_URL}/api/ps",
		)
		return [model["name"] for model in json.loads(response)["models"]]
	except Exception as e:
		logger.error(f"get_running_models - Error - {status_code} - {response} - {e}")

# returns a bool
async def load_model(model: str) -> Any:
	status_code, response = None, None
	try:
		status_code, response = await make_request(
			url=f"{OLLAMA_BASE_URL}/api/generate",
			is_post=True,
			data={
				"model" : model
			}
		)
		return json.loads(response)["done"]
	except Exception as e:
		logger.error(f"load_model - Error - {status_code} - {response} - {e}")

def whisper_load(model_path: str) -> Any:
	data = {"model": model_path}
	status_code, response = None, None
	try:
		logger.info(f"whisper_load - Making new request - {data}")
		response = requests.post(
			f"http://127.0.0.1:{WHISPER['port']}/load", 
			files=data
		)
		logger.info(f"whisper_load - Received response - {response.status_code} - {response.text}")
		return response.status_code, response.text
	except Exception as e:
		logger.error(f"whisper_load - Error - {status_code} - {response} - {e}")
		return status_code, response

def whisper_transcribe(file_path: str) -> Any:
	data = {
		"file": open(file_path, "rb"),
		"response_format": "json"
	}
	status_code, response = None, None
	try:
		load_status_code, load_response = whisper_load(WHISPER["model_path"])
		if load_status_code != 200:
			status_code, response = load_status_code, load_response
			raise Exception("whisper load model error")
		logger.info(f"whisper_transcribe - Making new request - {data}")
		response = requests.post(
			f"http://127.0.0.1:{WHISPER['port']}/inference", 
			files=data
		)
		logger.info(f"whisper_transcribe - Received - {response.status_code} - {response.text}")
		whisper_load(WHISPER["default_model_path"])
		status_code, json_response = response.status_code, response.json()
		logger.info(f"whisper_transcribe - Received json - {status_code} - {json_response}")
		return (status_code, "".join(json_response["text"].strip().splitlines())) if status_code == 200 else (status_code, response.text)
	except Exception as e:
		logger.error(f"whisper_transcribe - Error - {status_code} - {response} - {e}")
		whisper_load(WHISPER["default_model_path"])
		return status_code, response
# END LLM FUNCTIONS

# START INTERFACING FUNCTIONS
async def ai_chat(prompt: str, chat_id: str) -> str:
	await create_new_chat(CURRENT_TEXT_CHAT_ID)
	chats = await get_chats_with_chat_id_text(chat_id)

	message_list = []
	if chats is not None:
		if len(chats) > 0:
			for chat in chats:
				message_list.append({"role": "user", "content": chat["user"]})
				message_list.append({"role": "assistant", "content": chat["assistant"]})
	
	message_list.append({"role": "user", "content": prompt})
	message_id = await insert_user_input(chat_id, prompt)
	return await insert_assistant_input(message_id, await chat(TEXT_MODEL, message_list))

async def ai_generate_text(prompt: str) -> str:
	return await generate(TEXT_MODEL, prompt)

async def print_chat_with_chat_id(chat_id: str) -> str:
	chats = await get_chats_with_chat_id_text(chat_id)

	if chats is None:
		logger.info(f"print_chat_with_chat_id - Empty chat - {chats}")
		return "Empty chat"

	if len(chats) <= 0:
		logger.info(f"print_chat_with_chat_id - Empty chat - {chats}")
		return "Empty chat"

	chat_message = "*Chats*\n"
	for chat in chats:
		chat_message = chat_message + f"User: `{prompt[1]}`\n\nAi: `{prompt[2]}`\n"
	return chat_string

async def get_chat_list_with_snippet():
	chats_chat_id = await get_all_chats_chat_id()

	if chats is None:
		logger.info(f"get_chat_list_with_snippet - No chats - {chats}")
		return "No chats"

	if len(chats) <= 0:
		logger.info(f"get_chat_list_with_snippet - No chats - {chats}")
		return "No chats"

	chat_list_message = "*History*\n"
	for chat_id in chats_chat_id:
		try:
			chat = await get_single_row_chats_with_chat_id(chat_id)
			if chat is not None:
				if len(chat) > 0:
					chat_list_message = chat_list_message + f"`{chat_id}`\n`{(chat['user'])[:20]}...`\n\n"
		except Exception as e:
			logger.error(f"get_chat_list_with_snippet - chat_id - {e}")
	return chat_list_message

async def get_help_message():
	return textwrap.dedent(r"""
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
	`/v` - to do voice
	
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

async def get_running_text_models():
	model_list = await get_running_models()
	message = "Running models\n"
	for model in model_list:
		message = message +  f"`{model}`\n"
	return message

async def get_text_models():
	model_list = await get_models()
	message = "Available models\n"
	for model in model_list:
		message = message +  f"`{model}`\n"
	return message

async def print_bot(message: str, markdown: bool = True) -> None:
	split_message = util.smart_split(message, chars_per_string=3000)
	try:
		for message in split_message:
			await bot.send_message(CHAT_ID, message, parse_mode=('Markdown' if markdown else None))
	except Exception as e:
		logger.error(f"print_bot - Error - {e}")

def use_cosyvoice(input_file: str, output_file: str, prompt: str) -> None:
	import sys
	sys.path.append('/run/media/mesh/git_wsp/CosyVoice/third_party/Matcha-TTS')
	sys.path.append('/run/media/mesh/git_wsp/CosyVoice')
	from cosyvoice.cli.cosyvoice import CosyVoice2
	from cosyvoice.utils.file_utils import load_wav
	import torchaudio
	logger.info(f"clone_voice - Starting clone - {output_file}")
	cosyvoice = CosyVoice2(
		'/run/media/mesh/git_wsp/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
		load_jit=False, load_trt=False, fp16=False
	)
	logger.info(f"clone_voice - Loaded model")
	prompt_speech_16k = load_wav(input_file, 16000)
	logger.info(f"clone_voice - Created prompt speech")
	for i, j in enumerate(cosyvoice.inference_instruct2(prompt, "a soft voice female", prompt_speech_16k, stream=False)):
		torchaudio.save(output_file, j['tts_speech'], cosyvoice.sample_rate)
	logger.info(f"clone_voice - Done cloning voice")
	logger.info(f"clone_voice - Sending voice - {output_file}")
	cosyvoice.quit()

def use_dia_huggingface(input_file: str, transcribed_audio: str, prompt: str) -> Any:
	logger.info(f"use_dia_huggingface - Conneting client - {HF_TOKEN} - {input_file} - {transcribed_audio} - {prompt}")
	client = gradio_client.Client("nari-labs/Dia-1.6B", hf_token=HF_TOKEN)
	logger.info(f"use_dia_huggingface - Sending request - {input_file}")
	response = client.predict(
		text_input=f"[S1] {prompt}",
		audio_prompt_input=gradio_client.handle_file(input_file),
		transcription_input=f"[S1] {transcribed_audio}",
		max_new_tokens=3072,
		cfg_scale=1,
		temperature=1.8,
		top_p=3.1,
		cfg_filter_top_k=50,
		speed_factor=1,
		api_name="/generate_audio"
	)
	logger.info(f"use_dia_huggingface - Received response - {input_file}")
	return response

async def clone_voice(input_file: str, prompt: str) -> None:
	output_file = f"{gen_uuid('clone')}.wav"
	if VOICE_MODEL.lower() == "cosyvoice":
		await asyncio.to_thread(use_cosyvoice, input_file, output_file, prompt)
		await bot.send_audio(CHAT_ID, telebot.types.InputFile(output_file), caption=prompt)
		logger.info(f"clone_voice - Sent voice - {output_file}")
	elif VOICE_MODEL.lower() == "nari-labs/dia-1.6b":
		status_code, transcribed_audio = await asyncio.to_thread(whisper_transcribe, input_file)
		logger.info(f"clone_voice - Transcribed audio - {input_file} - {status_code} - {transcribed_audio}")
		if status_code != 200:
			logger.info(f"clone_voice - Could not transcribe audio input invalid status code - {input_file} - {status_code}")
			await print_bot(f"Could not transcribe audio input invalid status code `{status_code}`")
			return
		response = await asyncio.to_thread(use_dia_huggingface, input_file, transcribed_audio, prompt)
		logger.info(f"clone_voice - Response voice received - {input_file} - {response}")
		if response is not None:
			shutil.copy(response, output_file)
			logger.info(f"clone_voice - Copied cloned voice file - {output_file}")
			await bot.send_audio(CHAT_ID, telebot.types.InputFile(response), caption=prompt)
			logger.info(f"clone_voice - Sent cloned voice - {response}")

@bot.message_handler(commands=['start'])
async def exec_cmd_start(message: telebot.types.Message):
	await print_bot(f"ChatTG `{VERSION}`")

@bot.message_handler(commands=['help', 'h'])
async def exec_cmd_help(message: telebot.types.Message):
	await print_bot(await get_help_message())

@bot.message_handler(commands=['newchat', 'nc'])
async def exec_cmd_newchat(message: telebot.types.Message):
	global CURRENT_TEXT_CHAT_ID
	CURRENT_TEXT_CHAT_ID = gen_uuid("chat")
	await print_bot(f"New chat id: `{CURRENT_TEXT_CHAT_ID}`")

@bot.message_handler(commands=['get_current_chat_id'])
async def exec_cmd_get_current_chat_id(message: telebot.types.Message):
	print_bot(f"Current chat id: `{CURRENT_TEXT_CHAT_ID}`")

@bot.message_handler(commands=['get_text_models'])
async def exec_cmd_get_text_models(message: telebot.types.Message):
	await print_bot(await get_text_models())

@bot.message_handler(commands=['get_running_models'])
async def exec_cmd_get_running_models(message: telebot.types.Message):
	await print_bot(await get_running_text_models())

@bot.message_handler(commands=['get_chat_list'])
async def exec_cmd_get_chat_list(message: telebot.types.Message):
	await print_bot(await get_chat_list_with_snippet())

@bot.message_handler(commands=['get_current_text_model'])
async def exec_cmd_get_current_text_model(message: telebot.types.Message):
	await print_bot(f"Current text model: `{TEXT_MODEL}`")

@bot.message_handler(commands=['ai'])
async def exec_cmd_ai(message: telebot.types.Message):
	await print_bot("Running...")
	await print_bot(await ai_chat(message.text.replace("/ai ", ""), CURRENT_TEXT_CHAT_ID))

@bot.message_handler(commands=['chat'])
async def exec_cmd_chat(message: telebot.types.Message):
	await print_bot("Running...")
	await print_bot(await ai_chat(message.text.replace("/chat ", ""), CURRENT_TEXT_CHAT_ID))

@bot.message_handler(commands=['generate'])
async def exec_cmd_generate(message: telebot.types.Message):
	await print_bot("Running...")
	await print_bot(await ai_generate_text(message.text.replace("/generate ", ""), CURRENT_TEXT_CHAT_ID))

@bot.message_handler(commands=['load_text_model'])
async def exec_cmd_load_text_model(message: telebot.types.Message):
	configs = await preload_config(CONFIG_DIRECTORY)
	configs["text"]["user_set_model"] = message.text.split()[1]
	await presave_config(CONFIG_DIRECTORY, configs)
	load_config(CONFIG_DIRECTORY)

	await print_bot("Loading...")
	await load_model(TEXT_MODEL)
	await print_bot(await get_running_text_models())

@bot.message_handler(commands=['set_text_model'])
async def exec_cmd_set_text_model(message: telebot.types.Message):
	configs = await preload_config(CONFIG_DIRECTORY)
	configs["text"]["user_set_model"] = message.text.split()[1]
	await presave_config(CONFIG_DIRECTORY, configs)
	load_config(CONFIG_DIRECTORY)
	await print_bot(f"Current text model: `{TEXT_MODEL}`")

@bot.message_handler(commands=['print_chat'])
async def exec_cmd_print_chat(message: telebot.types.Message):
	await print_bot(await print_chat_with_chat_id(message.text.split()[1]))

@bot.message_handler(commands=['select_chat'])
async def exec_cmd_select_chat(message: telebot.types.Message):
	global CURRENT_TEXT_CHAT_ID
	CURRENT_TEXT_CHAT_ID = message.text.split()[1]
	await print_bot(f"Current chat id: `{CURRENT_TEXT_CHAT_ID}`")

@bot.message_handler(content_types=['audio', 'photo', 'voice', 'video', 'document',
    'text', 'location', 'contact', 'sticker'])
async def handle_file(message: telebot.types.Message):
	try:
		logger.info(f"handle_file - New file request received - {message.caption}")

		if str(message.chat.id) != CHAT_ID:
			await print_main("*Invalid chat id*")
			return

		if message.content_type == "audio":
			logger.info(f"handle_file - Received audio")
			telegram_file_path = await bot.get_file(message.audio.file_id)
			logger.info(f"handle_file - File path - {telegram_file_path}")
			file_data = await bot.download_file(telegram_file_path.file_path)
			logger.info(f"handle_file - File data")
			if str(message.caption).startswith("/v"):
				raw_file = f"/tmp/{gen_uuid('raw')}"
				wav_file = f"/tmp/{gen_uuid('wav')}"
				with open(raw_file, "wb") as input_file:
					input_file.write(file_data)
				sound = AudioSegment.from_mp3(raw_file)
				sound.export(wav_file, format="wav")
				logger.info(f"handle_file - Running...")
				await print_bot("Running...")
				prompt = str(message.caption).replace("/v ", "")
				await clone_voice(wav_file, prompt)
		else:
			telegram_file_path = await bot.get_file(message.document.file_id)
			file_data = await bot.download_file(telegram_file_path.file_path)
			await print_bot("Invalid type")
			return
		logger.info(f"handle_file - handled file")
	except Exception as e:
		logger.error(f"handle_file - Error - {e}")
		await print_bot("*Error parsing file*")

async def main():
	global CURRENT_TEXT_CHAT_ID, aiohttp_session
	logger.info("main - Starting ai_linux")
	logger.info("main - Initialized bot")
	aiohttp_session = aiohttp.ClientSession()
	logger.info("main - Created aiohttp session")
	await _mkdir(f"{WORKING_DIRECTORY}/databases")
	logger.info("main - Created database directory")
	CURRENT_TEXT_CHAT_ID = gen_uuid("chat")
	await init_database(CHAT_DATABASE_DIRECTORY)
	logger.info("main - Initialized database")
	logger.info("main - Started polling")
	await bot.infinity_polling()

if __name__ == "__main__":
	asyncio.run(main())