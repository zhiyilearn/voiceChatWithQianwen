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

class llmchatwithQianWenSystem:
    def __init__(self):
        # LLM Initialization
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Vosk Speech Recognition Initialization
        try:
            self.model = Model("../models/vosk-model-small-cn-0.3")
        except Exception as e:
            print(f"Vosk Model Error: {e}")
            sys.exit(1)
        self.recognizer = KaldiRecognizer(self.model, 16000)  # 16kHz sampling rate
        
        # Microphone Stream (persistent - avoid reinitializing)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096
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
        completion = self.client.chat.completions.create(
            model="qwen3-vl-flash",
            messages=[{
                'role': 'user',
                'content': text
            }]
        )
        return completion.choices[0].message.content

    # ---------------------- Chinese TTS Setup (Unchanged) ----------------------
    def setup_chinese_tts(self):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for v in voices:
            if 'cmn-latn-pinyin' in v.languages:
                engine.setProperty('voice', v.id)
                break
        engine.setProperty('rate', 150)  # Speech speed
        engine.setProperty('volume', 0.9)  # Volume
        return engine

    # ---------------------- Wakeup (Modified for Interruption) ----------------------
    def wakeup(self):
        self.input_text = "好"
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
        data = self.stream.read(4096, exception_on_overflow=False)
        
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
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
                data = self.stream.read(4096, exception_on_overflow=False)
                
                # Check for partial voice (real-time detection)
                if not self.recognizer.AcceptWaveform(data):
                    partial = json.loads(self.recognizer.PartialResult())
                    if partial["partial"].strip():
                        self.interrupt_tts.set()  # Trigger interruption
                
                # Check for complete voice (fallback)
                else:
                    result = json.loads(self.recognizer.Result())
                    if result["text"].strip():
                        self.interrupt_tts.set()  # Trigger interruption
            
            time.sleep(0.05)  # Reduce CPU usage

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
        print("\n👋 Program exited by user")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

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
