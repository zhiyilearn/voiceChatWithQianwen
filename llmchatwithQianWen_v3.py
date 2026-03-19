# Version 3. Add speech recognition function
from openai import OpenAI
import os
import sys
import vosk
import pyttsx3

import numpy as np
import threading
import time
import signal
import sys

client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

'''
# (1) Define memory
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

 
# (2) Contruct State Machine Arch
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)

# (3) Qianwen model 
from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(
    model = "qwen3-vl-flash",
    streaming=True,
    temperature=0.7, 
    top_p= 0.8, 
    # api_key = "sk-47b5da865fa04a6ca3c3f9b19485727d",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 4. Node for chat process
def get_response(text):
    client = OpenAI(
        api_key = os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
         model="qwen3-vl-flash",
         messages=[
              {
                  'role': 'user',
                  'content': text
              }
         ])

def chatbot(state: State):
    return {"messages" : get_response("messages")}

graph_builder.add_node("chatbot", chatbot)

# 5. Connect the flowchat
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 6. Compile State Machine
graph = graph_builder.compile()

# 7 Implement stream chat
def stream_graph_update(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1])


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_update(user_input)
    except:
        # Assume input() is not available
        user_input ="How are you?"
        print("User: " + user_input)
        stream_graph_update(user_input)
        break



def say_chinese(text):
    engine = pyttsx3.init()

    # 示例（需系统安装中文语音）
    voices = engine.getProperty("voices")
    # for voice in voices:
    # chinese_voice = next(v for v in voices if "zh" in v.languages[0])
    # engine.setProperty('voice', 'zh')

    engine.setProperty('rate', 150) # Adjust speed
    engine.setProperty('volume', 0.9) 


    engine.say(text)
    engine.runAndWait()
    # engine.stop()  
'''


# 创建聊天完成请求
def get_response(text):
    completion = client.chat.completions.create(
         model="qwen3-vl-flash",
         messages=[
              {
                  'role': 'user',
                  'content': text
              }
         ])
    return completion.choices[0].message.content

def setup_chinese_tts():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for v in voices:
        # if 'zh' in v.languages or 'chinese' in v.name.lower():
        if 'cmn-latn-pinyin' in v.languages:
            print(v.languages)
            print(v.name)
            engine.setProperty('voice', v.id)
            break
    return engine
 

# Wake up/Initiate the conversation.
# input_text = 'How are you? '
input_text = "好"
engine = setup_chinese_tts()
engine.say(input_text)
engine.runAndWait()


# say_chinese(input_text)
# print(input_text)
# print("Stop.")

# sys.exit(0)

# Process and wait speaker finish
print(input_text)

# Create a recognizer object
from vosk import Model, KaldiRecognizer
import pyaudio
import json
# 初始化模型（需指定模型路径）
model = Model("../models/vosk-model-small-cn-0.3")
recognizer = KaldiRecognizer(model, 16000)  # 采样率16kHz

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1,
                rate=16000, input=True, frames_per_buffer=4096)


while True:
    # User voice
    input_text = ""
    data = stream.read(4096)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        input_text = result["text"]
        print("识别结果:", result["text"])
        
    '''
    else:
        partial = json.loads(recognizer.PartialResult())
        print("实时结果:", partial["partial"])
    '''
    # input_text = input("User: ")
    if input_text.lower() in ["quit", "exit", "q"]:
        engine = setup_chinese_tts()
        engine.say("GoodByle!")
        engine.runAndWait()

        # say_chinese("GoodByle!")

        # Process and wait speaker finish
        print("Goodbye!")

        break

    # LLM
    if input_text != "":
        response_text = get_response(input_text)
        engine = setup_chinese_tts()
        engine.say(response_text)
        engine.runAndWait()
        
        # say_chinese(response_text)
        # Process and wait speaker finish
        print(response_text)



