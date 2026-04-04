# Version 10: Use Whisper for asr 
from openai import OpenAI
import os
import sys
import pyttsx3
import numpy as np
import threading
import time
import signal
import json
import pyaudio
import subprocess
import tempfile
import wave
import webrtcvad

import whisper
# 启用镜像（自动走国内CDN）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Audio recording parametersls
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono audio
RATE = 16000              # 16kHz sample rate (good for speech)
CHUNK = 512               # Buffer size
RECORD_SECONDS = 10         # Duration of recording
TEMP_WAV = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

# For chunk separation
import re

# For human voice detection
# Audio Configuration (optimized for voice detection)
# --------------------------
CHUNK_DURATION_MS = 30  # 30ms chunks (required for VAD)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)


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
        
        # Whisper Speech Recognition Initialization
        try:
            print("🧠 Loading Whisper model (base)...")
            self.whisper_model = whisper.load_model("tiny")  # Options: tiny, base, small, medium, largeself
        except Exception as e:
            print(f"Whisper Model Error: {e}")
            sys.exit(1)
        
        # ---------------------- Critical Fix: Separate Vosk Recognizers ----------------------
              
        # Microphone Stream (persistent - avoid reinitializing)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            # frames_per_buffer=CHUNK_SIZE
            frames_per_buffer=512
        )

        # Initialize VAD (Voice Activity Detector)
        # Aggressiveness: 0-3 (3 = most strict, filters more noise)
        self.vad = webrtcvad.Vad(3)

        # Handle microphone recording and AST with whisper
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
        # engine.setProperty('voice', 'zh+f3')  
        engine.setProperty('rate', 200)  # Speech speed
        engine.setProperty('volume', 1.0)  # Volume
        return engine

    # ---------------------- Wakeup (Modified for Interruption) ----------------------
    def wakeup(self):
        self.input_text = "你好"
        engine = self.setup_chinese_tts()
        # Use interruptible TTS for wakeup
        engine.say(self.input_text)
        engine.runAndWait()
        engine.stop()
        return self.input_text

    def split_text_by_separators(self, text: str) -> list:
        """
        Split text into chunks using separators: , . ? !
        Preserves the separator at the end of each chunk
        """
        # Regex pattern: split AFTER any of , . ? !
        pattern = r'(?<=[,.?!])|\n'

        # Split text and clean empty/whitespace chunks
        chunks = re.split(pattern, text)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        print(chunks)
        return chunks

    # ---------------------- Core: Interruptible TTS Playback ----------------------
    def _speak_with_interrupt(self, text, engine):
        """Internal method: speak text with real-time interruption support"""
        self.tts_playing = True
        self.interrupt_tts.clear()  # Reset interrupt flag

        # Split Chinese text into small chunks for granular interruption
        # chunks = text.split('，') if '，' in text else text.split('。')
        #chunks = text.split('\n') if '\n' in text else text.split('。')
        # Split Chinese text into small chunks based on separator , ? .
        chunks = self.split_text_by_separators(text)
        if len(chunks) == 1:
            chunks = list(text)  # Fallback to individual characters

        # Play chunk by chunk (check interrupt flag each time)
        for chunk in chunks:
            # if self.interrupt_tts.is_set() or not chunk.strip():
            if self.interrupt_tts.is_set():
                print("\n🔴 Speaker interrupted by your voice!")
                break  # Stop immediately if interrupt is triggered
            engine.say(chunk)
            engine.runAndWait()  # Play only one small chunk

        self.tts_playing = False
        engine.stop()
        if self.interrupt_tts.is_set():
            self.interrupt_tts.clear()  # Reset flag after interruption

    # ---------------------- Modified LLM Talk (Interruptible) ----------------------
    def llm_talk(self, text):
        engine = self.setup_chinese_tts()
        self._speak_with_interrupt(text, engine)  # Use interruptible playback

   

    # ---------------------- Microphone Input (Refactored) ----------------------
    def record_audio(self, filename):
        print("🎤 Recording... Speak now!")
        frames = []
        
        try:
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            print("✅ Recording finished.")
            
        # Save the recorded audio to a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    def transcribe_audio(self, filename):
        """Transcribe audio file using Whisper."""
        input_text = ""
        try:
            #print("🧠 Loading Whisper model (base)...")
            # model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
            print("🔍 Transcribing...")
            result = self.whisper_model.transcribe(filename)
            print("\n📝 Transcription:")
            print(result["text"])
            input_text = result["text"]
        except Exception as e:
            print(f"Error during transcription: {e}")
    
        return input_text
    
    def getFromMicro(self):
        self.input_text = ""
        try:
            self.record_audio(TEMP_WAV)
            self.input_text = self.transcribe_audio(TEMP_WAV)
        finally:
            # Clean up temporary file
            if os.path.exists(TEMP_WAV):
                os.remove(TEMP_WAV)

        return self.input_text


    def is_human_voice(self, audio_chunk):
        """Verify if audio chunk contains human voice (returns True/False)"""
        # Convert raw audio to bytes (VAD input format)
        return self.vad.is_speech(audio_chunk, RATE)
    
    # ---------------------- Background: Microphone Interrupt Monitor ----------------------
    def _mic_interrupt_monitor(self):
        """Background thread: monitor mic for voice to interrupt TTS"""
        print("\n🎤 Microphone monitor started (speak to interrupt speaker)...")
        while True:
            # Only monitor when TTS is playing (save CPU)
            if self.tts_playing:
                raw_data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Check for human voice
                voice_detected = self.is_human_voice(raw_data)
        
                 # Print result
                if voice_detected:
                    print("HUMAN VOICE DETECTED")
                    self.interrupt_tts.set() 
                else:
                    print("Silence / Noise")
                    # self.interrupt_tts.clear() 
                
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
        # Get new voice input from microphone
        input_text = chat_System.getFromMicro()

        if input_text.strip():
            try:
                response_text = chat_System.get_response(input_text)
                # Filter out non-chinese character **
                # res_str = str.replace('e', '') 
                filtered_response_text = response_text.replace('*', '')

                print(f"Assistant: {filtered_response_text}")
                chat_System.llm_talk(filtered_response_text)
            except Exception as e:
                print(f"LLM Error: {e}")
                response_text = "抱歉，我暂时无法回答你的问题"
                chat_System.llm_talk(response_text)
        
        # Exit condition
        if input_text.lower() in ["quit", "exit", "q", "退出", "结束"]:
            chat_System.llm_talk("再见！祝你生活愉快！")
            chat_System.cleanup()
            break
