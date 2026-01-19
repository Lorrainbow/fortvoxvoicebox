import pyaudio, wave, json, os, sys, numpy as np
from time import sleep
from gpiozero import Button, Servo
from vosk import Model, KaldiRecognizer, SpkModel
from demo_opts import get_device
from luma.core.render import canvas
from PIL import ImageFont
from pathlib import Path

# -------------------------
# Hardware
# -------------------------
THRESHOLD = 0.5
PASSWORD_LENGTH = 5
user_button = Button(16)

record_button = Button(17)
servo = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0024)
CHANNELS = 1
RATE = 16000
CHUNK = 1024
INPUT_DEVICE = None
OUTPUT_DEVICE = 2

# find ReSpeaker input
p = pyaudio.PyAudio()
for i in range(p.get_host_api_info_by_index(0)['deviceCount']):
    info = p.get_device_info_by_host_api_device_index(0, i)
    if info.get('maxInputChannels')>0 and info['name'].startswith("seeed"):
        INPUT_DEVICE=i
        break
p.terminate()
if INPUT_DEVICE is None: sys.exit("ReSpeaker input not found")

# -------------------------
# Servo helpers
# -------------------------
def open_servo():
    servo.value=-1; sleep(1); servo.value=1; sleep(1)
def close_servo():
    servo.value=1; sleep(1); servo.value=-1; sleep(1)
close_servo()

# -------------------------
# Screen / font
# -------------------------
device = get_device()
font = ImageFont.truetype(str(Path(__file__).parent.joinpath('fonts','FreePixel.ttf')), device.height-40)
def draw_text(msg):
    with canvas(device) as draw: draw.text((0,4), msg, fill="white", font=font)

# -------------------------
# Audio helpers
# -------------------------
def record_wav(filename, duration=PASSWORD_LENGTH):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                    input=True, input_device_index=INPUT_DEVICE, frames_per_buffer=CHUNK)
    frames=[stream.read(CHUNK, exception_on_overflow=False) for _ in range(int(RATE/CHUNK*duration))]
    stream.stop_stream(); stream.close(); p.terminate()
    wf = wave.open(filename,'wb'); wf.setnchannels(CHANNELS); wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE); wf.writeframes(b''.join(frames)); wf.close()
    #draw_text("Calculating...")
    return frames

def play_wav(filename):
    draw_text("Playing...")
    wf = wave.open(filename,'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True, output_device_index=OUTPUT_DEVICE)
    data = wf.readframes(CHUNK)
    while data: stream.write(data); data = wf.readframes(CHUNK)
    stream.stop_stream(); stream.close(); p.terminate(); wf.close()

# -------------------------
# Vosk helpers
# -------------------------
def cosine_dist(x,y): return 1 - np.dot(np.array(x),np.array(y))/np.linalg.norm(x)/np.linalg.norm(y)
def get_signature(wav_file, rec):
    wf = wave.open(wav_file,'rb') if isinstance(wav_file,str) else wav_file
    while True:
        data = wf.readframes(4000)
        if len(data)==0: break
        if rec.AcceptWaveform(data):
            res=json.loads(rec.Result()); 
            if "spk" in res: return res["spk"]
    return json.loads(rec.FinalResult()).get("spk", None)

# -------------------------
# Vosk model
# -------------------------
SPK_MODEL_PATH="/home/cloud/vosk-model-spk-0.4"
if not os.path.exists(SPK_MODEL_PATH): draw_text("No model"); sys.exit(1)
model = Model(lang="en-us"); spk_model = SpkModel(SPK_MODEL_PATH)
rec = KaldiRecognizer(model, RATE); rec.SetSpkModel(spk_model)

# -------------------------
# Load password
# -------------------------
PASSWORD_FILE="password.wav"
if not os.path.exists(PASSWORD_FILE):
    draw_text("No password file"); sys.exit(1)
password_sig = get_signature(PASSWORD_FILE, rec)
draw_text("Ready")

# -------------------------
# Main loop
# -------------------------
while True:
    if record_button.is_pressed:
        draw_text("Recording...")
        record_wav(PASSWORD_FILE)
        password_sig=get_signature(PASSWORD_FILE, rec)
        play_wav(PASSWORD_FILE)
        draw_text("Ready")
        sleep(1)

    if user_button.is_pressed:
        play_wav(PASSWORD_FILE)
        draw_text("Speak")
        ATTEMPT_FILE="attempt.wav"
        record_wav(ATTEMPT_FILE)
        #if you want to play their attempt back, uncomment next line
        #play_wav(ATTEMPT_FILE)
        
        attempt_sig=get_signature(ATTEMPT_FILE, rec)
        if attempt_sig is None: draw_text("Silence"); continue
        diff = cosine_dist(password_sig, attempt_sig)
        draw_text(f"Score: {diff:.2f}")
        sleep(2)
        if diff<THRESHOLD:
            draw_text("You win!"); open_servo(); sleep(2); close_servo()
        else:
            draw_text("Try again!")
        sleep(2); draw_text("Ready")
