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

from vosk import Model, KaldiRecognizer
import pyaudio
import json

class llmchatwithQianWenSystem:
    def __init__(self):
        self.client = OpenAI(
                    api_key = os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 初始化模型（需指定模型路径）
        self.model = Model("../models/vosk-model-small-cn-0.3")
        self.recognizer = KaldiRecognizer(self.model, 16000)  # 采样率16kHz        

    # 创建聊天完成请求
    def get_response(self, text):
        completion = self.client.chat.completions.create(
                    model="qwen3-vl-flash",
                    messages=[
                    {
                        'role': 'user',
                        'content': text
                    }
         ])
        return completion.choices[0].message.content

    def setup_chinese_tts(self):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for v in voices:
             # if 'zh' in v.languages or 'chinese' in v.name.lower():
            if 'cmn-latn-pinyin' in v.languages:
            #print(v.languages)
            # print(v.name)
                engine.setProperty('voice', v.id)
                break
        return engine
 
    def wakeup(self):
        # Wake up/Initiate the conversation.
        # input_text = 'How are you? '
        self.input_text = "好"
        engine = self.setup_chinese_tts()
        engine.say(self.input_text)
        engine.runAndWait()
        engine.stop()

        return self.input_text


    def getFromMicro(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1,
                rate=16000, input=True, frames_per_buffer=4096)
        self.input_text = ""
        data = stream.read(4096)
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            self.input_text = result["text"]
            print("识别结果:", result["text"])
        
        return self.input_text

    def llm_talk(self, text):
        engine = self.setup_chinese_tts()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

if __name__ == "__main__":
    chat_System = llmchatwithQianWenSystem()
    input_text = chat_System.wakeup()
    print(input_text)

    while True:
        # LLM
        if input_text != "":
            response_text = chat_System.get_response(input_text)
            print(response_text)
            chat_System.llm_talk(response_text)
            

        input_text = chat_System.getFromMicro()
        print(input_text)

        # break
    