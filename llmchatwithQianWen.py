# Version 10: Use Whisper for asr 
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

import tempfile
import os
import sys
import wave

import whisper
import os
# 启用镜像（自动走国内CDN）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Audio recording parametersls
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono audio
RATE = 16000              # 16kHz sample rate (good for speech)
CHUNK = 512               # Buffer size
RECORD_SECONDS = 10         # Duration of recording
TEMP_WAV = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

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

        # Vosk Automatic Speech Synthesis Initialization
        try:
            self.vosk_model = Model("../models/vosk-model-small-cn-0.3")
        except Exception as e:
            print(f"Vosk Model Error: {e}")
            sys.exit(1)
        
        # Whisper Speech Recognition Initialization
        try:
            print("🧠 Loading Whisper model (base)...")
            self.whisper_model = whisper.load_model("tiny")  # Options: tiny, base, small, medium, largeself
        except Exception as e:
            print(f"Whisper Model Error: {e}")
            sys.exit(1)
        
        # ---------------------- Critical Fix: Separate Vosk Recognizers ----------------------
        
        # Recognizer 2: For interrupt detection (background thread)
        self.recognizer_interrupt = KaldiRecognizer(self.vosk_model, 16000)

              
        # Microphone Stream (persistent - avoid reinitializing)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=512
        )

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
        self.input_text = "你好，我已准备好与你对话"
        engine = self.setup_chinese_tts()
        # Use interruptible TTS for wakeup
        self._speak_with_interrupt(self.input_text, engine)
        return self.input_text

    # ---------------------- Core: Interruptible TTS Playback ----------------------
    def _speak_with_interrupt(self, text, engine):
        """Internal method: speak text with real-time interruption support"""
        self.tts_playing = True
        # self.interrupt_tts.clear()  # Reset interrupt flag

        # Split Chinese text into small chunks for granular interruption
        chunks = text.split('，') if '，' in text else text.split('。')
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
                        print(partial["partial"])
                        self.interrupt_tts.set()  # Trigger interruption
                
                # Check for complete voice (fallback)
                else:
                    result = json.loads(self.recognizer_interrupt.Result())
                    if result["text"].strip():
                        print(result["text"])
                        # rms_amplitude = np.sqrt(np.mean(np.square(audio_data))) / 32768.0 
                        # print(rms_amplitude)
                        # if rms_amplitude > 0.1:
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
