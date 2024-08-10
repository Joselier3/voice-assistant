import time
import wave
import queue
import struct
import threading
import subprocess

import pyaudio
from tools.audiofile import AudioFile

from openai import OpenAI
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEBUG = True

# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * RATE / CHUNK  # two seconds of silence marks the end of user voice input
MIC_IDX = 1 # Set microphone id. Use tools/list_microphones.py to see a device list.

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('recordings/output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False

    def on_llm_end(self, response, **kwargs):
        # print("on_llm_new_token() called")
        # Append the token to the generated text
        with self.lock:
            self.speech_queue.put(response.generations[0][0].text)

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        try:
            print("Creating audio")
            time_ckpt = time.time()
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=text,
                response_format='wav'
            ) as response:
                response.stream_to_file("responses/output.wav")


            print("Reproducing audio file (Time %d ms)" % ((time.time() - time_ckpt) * 1000))
            audio_file = AudioFile("responses/output.wav")
            audio_file.play()
            audio_file.close()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


if __name__ == '__main__':
    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    dialogue = [{"role": "system", "content": "You're a helpful server in a waffle shop in Santo Domingo."}]
    try:
        while True:
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("Listening...")
                record_audio()
                print("Transcribing...")
                time_ckpt = time.time()

                client = OpenAI(api_key=OPENAI_API_KEY)
                audio_file = open("recordings/output.wav", "rb")
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
                user_input = transcription.text
                print("%s: %s (Time %d ms)" % ("Guest", user_input, (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("voice recognition failed, please try again")
                continue
            time_ckpt = time.time()
            print("Generating...")
            dialogue.append({"role": "user", "content": user_input})

            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=GROQ_API_KEY,
                callbacks=[voice_output_handler]
            )
            
            reply = llm.invoke(dialogue)
            dialogue.append({"role": "assistant", "content": reply.content})

            if reply is not None:
                voice_output_handler.speech_queue.put(None)
                print("%s: %s (Time %d ms)" % ("Server", reply.content, (time.time() - time_ckpt) * 1000))
    except KeyboardInterrupt:
        pass
