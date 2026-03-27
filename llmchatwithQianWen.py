# Version 9: Add context related functions 
from openai import OpenAI
import os
import sys
import vosk
import pyttsx3
import numpy as np
import threading
import time
import signal
import json
import pyaudio

from vosk import Model, KaldiRecognizer
import pyaudio
import json
import subprocess

import noisereduce as nr

class llmchatwithQianWenSystem:
    def __init__(self):
        self.restart_audio()

        # LLM
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # ====================== ADD CONTEXT (MEMORY) HERE ======================
        self.conversation_history = []
        # Optional: Set system prompt to define AI behavior
        self.conversation_history.append({
            "role": "system",
            "content": "你是一个有用的中文智能助手，回答简洁、准确、友好。"
        })

        # Vosk Speech Recognition Initialization
        try:
            self.model = Model("../models/vosk-model-small-cn-0.3")
        except Exception as e:
            print(f"Vosk Model Error: {e}")
            sys.exit(1)
        
        # ---------------------- Critical Fix: Separate Vosk Recognizers ----------------------
        # Recognizer 1: For user speech recognition (main thread)
        self.recognizer_recognition = KaldiRecognizer(self.model, 16000)
        # Recognizer 2: For interrupt detection (background thread)
        self.recognizer_interrupt = KaldiRecognizer(self.model, 16000)

        # Used to re-start audio function
        # self.restart_audio()
        
        # Microphone Stream (persistent - avoid reinitializing)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=512
        )

        # Thread-Safe Interrupt Flags (core for speaker interruption)
        self.interrupt_tts = threading.Event()  # Interrupt trigger flag
        self.tts_playing = False  # Track if TTS is active

        # Start Microphone Interrupt Monitor (background thread)
        self.monitor_thread = threading.Thread(
            target=self._mic_interrupt_monitor,
            daemon=True  # Auto-terminate with main program
        )
        self.monitor_thread.start()
 
    # ---------------------- LLM Response (Unchanged) ----------------------
    def get_response(self, text):
        # Add user input to history
        self.conversation_history.append({
            "role": "user",
            "content": text
        })

        # Send FULL HISTORY to QianWen API
        completion = self.client.chat.completions.create(
            model="qwen3-vl-flash",
            messages=self.conversation_history  # ✅ CONTEXT HERE
        )

        # Get AI reply
        ai_reply = completion.choices[0].message.content

        # Add AI reply to history (so it remembers next time)
        self.conversation_history.append({
            "role": "assistant",
            "content": ai_reply
        })

        return ai_reply
    # ---------------------- Chinese TTS Setup (Unchanged) ----------------------
    def setup_chinese_tts(self):
        engine = pyttsx3.init(driverName='espeak')
        voices = engine.getProperty('voices')
        for v in voices:
            if 'cmn-latn-pinyin' in v.languages:
                engine.setProperty('voice', v.id)
                break
        engine.setProperty('rate', 200)  # Speech speed
        engine.setProperty('volume', 1.0)  # Volume
        return engine

    # ---------------------- Wakeup (Modified for Interruption) ----------------------
    def wakeup(self):
        self.input_text = "你好，我已准备好与你对话"
        engine = self.setup_chinese_tts()
        # Use interruptible TTS for wakeup
        self._speak_with_interrupt(self.input_text, engine)
        return self.input_text

    # ---------------------- Core: Interruptible TTS Playback ----------------------
    def _speak_with_interrupt(self, text, engine):
        """Internal method: speak text with real-time interruption support"""
        self.tts_playing = True
        self.interrupt_tts.clear()  # Reset interrupt flag

        # Split Chinese text into small chunks for granular interruption
        chunks = text.split('，') if '，' in text else text.split('。')
        if len(chunks) == 1:
            chunks = list(text)  # Fallback to individual characters

        # Play chunk by chunk (check interrupt flag each time)
        for chunk in chunks:
            if self.interrupt_tts.is_set() or not chunk.strip():
                break  # Stop immediately if interrupt is triggered
            engine.say(chunk)
            engine.runAndWait()  # Play only one small chunk

        self.tts_playing = False
        engine.stop()
        if self.interrupt_tts.is_set():
            print("\n🔴 Speaker interrupted by your voice!")
            self.interrupt_tts.clear()  # Reset flag after interruption

    # ---------------------- Modified LLM Talk (Interruptible) ----------------------
    def llm_talk(self, text):
        engine = self.setup_chinese_tts()
        self._speak_with_interrupt(text, engine)  # Use interruptible playback

    # ---------------------- Microphone Input (Refactored) ----------------------
    def getFromMicro(self):
        """Get voice input from microphone (non-blocking)"""
        self.input_text = ""
        raw_data = self.stream.read(512, exception_on_overflow=False)

        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=16000)
        cleaned_data = reduced_noise.tobytes()
        
         # Use dedicated recognizer for speech recognition
        if self.recognizer_recognition.AcceptWaveform(cleaned_data):
            result = json.loads(self.recognizer_recognition.Result())
            self.input_text = result["text"].strip()
            if self.input_text:
                print("识别结果:", self.input_text)

        return self.input_text

    # ---------------------- Background: Microphone Interrupt Monitor ----------------------
    def _mic_interrupt_monitor(self):
        """Background thread: monitor mic for voice to interrupt TTS"""
        print("\n🎤 Microphone monitor started (speak to interrupt speaker)...")
        while True:
            # Only monitor when TTS is playing (save CPU)
            if self.tts_playing:
                raw_data = self.stream.read(512, exception_on_overflow=False)

                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                reduced_noise = nr.reduce_noise(y=audio_data, sr=16000)
                cleaned_data = reduced_noise.tobytes()

                 # Use dedicated recognizer for interrupt detection
                if not self.recognizer_interrupt.AcceptWaveform(cleaned_data):
                    partial = json.loads(self.recognizer_interrupt.PartialResult())
                    if partial["partial"].strip():
                        self.interrupt_tts.set()  # Trigger interruption
                
                # Check for complete voice (fallback)
                else:
                    result = json.loads(self.recognizer_interrupt.Result())
                    if result["text"].strip():
                        self.interrupt_tts.set()  # Trigger interruption
                        
            # Reduce CPU usage
            time.sleep(0.001)

    def restart_audio(self):
        try:
            # Executes the command and waits for it to complete
            subprocess.run(
            ["systemctl", "--user", "restart", "pipewire", "pipewire-pulse"],
            check=True
            )
            print("Audio services restarted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to restart audio services: {e}")

    # ---------------------- Cleanup (Added) ----------------------
    def cleanup(self):
        """Clean up resources on exit"""
        self.interrupt_tts.set()  # Stop any ongoing TTS
        time.sleep(0.1)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
        print("\n🧹 Resources cleaned up successfully")

        
# ---------------------- Main Program ----------------------
if __name__ == "__main__":
    # Initialize chat system
    chat_System = llmchatwithQianWenSystem()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        chat_System.cleanup()
        print("\n👋 退出")
        sys.exit(0)
        
    # Wakeup the system
    input_text = chat_System.wakeup()
    print(f"Assistant: {input_text}")

    # Main conversation loop
    while True:
        # Get LLM response and speak (interruptible)
        if input_text.strip():
            try:
                response_text = chat_System.get_response(input_text)
                print(f"Assistant: {response_text}")
                chat_System.llm_talk(response_text)
            except Exception as e:
                print(f"LLM Error: {e}")
                response_text = "抱歉，我暂时无法回答你的问题。"
                chat_System.llm_talk(response_text)
        
        # Get new voice input from microphone
        input_text = chat_System.getFromMicro()
        
        # Exit condition
        if input_text.lower() in ["quit", "exit", "q", "退出", "结束"]:
            chat_System.llm_talk("再见！祝你生活愉快！")
            chat_System.cleanup()
            break
