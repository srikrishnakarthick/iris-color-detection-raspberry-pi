import pygame
import numpy as np
import time
from pynput import keyboard

# ---------------- AUDIO INIT ----------------
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

SAMPLE_RATE = 44100
SA = 277.18 #D#

# ---------------- SWARA SYSTEM ----------------
SHUDDHA = {
    "sa": 1, "re": 9/8, "ga": 5/4,
    "ma": 4/3, "pa": 3/2,
    "dha": 5/3, "ni": 15/8
}

KOMAL = {
    "re": 16/15, "ga": 6/5,
    "dha": 8/5, "ni": 9/5
}

TIVRA = {
    "ma": 45/32
}

# ---------------- RAGA RULES ----------------
RAGA_RULES = {
    "NORMAL": {},
    "KAFI": {"ga": "komal", "ni": "komal"},
    "MULTANI": {"re": "komal", "ga": "komal", "dha": "komal", "ma": "tivra"},
    "AHEERBHAIRAV": {"re": "komal", "ni": "komal"},
    "BHIMPALASI": {"ga": "komal", "ni": "komal"},
    "YAMAN": {"ma": "tivra"},
    "BHAIRAV": {"re": "komal", "dha": "komal"},
    "BHAIRAVI": {"re": "komal", "ga": "komal", "dha": "komal", "ni": "komal"},
    "MALKAUNS": {"ga": "komal", "dha": "komal", "ni": "komal"},
    "CHANDRAKAUNS": {"ga": "komal", "dha": "komal"},
    "ABHOGIKANADA": {"ga": "komal"},
}

# ---------------- RAGA SELECTION ----------------
print("\n🎼 AVAILABLE RAGAS:")
for r in RAGA_RULES:
    print(" -", r)

RAGA = input("\n👉 Enter raga name: ").strip().upper()

if RAGA not in RAGA_RULES:
    print("⚠ Invalid raga! Using NORMAL.")
    RAGA = "NORMAL"

print(f"\n🎶 Selected Raga: {RAGA}\n")

# ---------------- KEY MAP ----------------
KEY_MAP = {
    'w': ('sa', 1),
    'e': ('re', 1),
    'f': ('ga', 1),
    't': ('ma', 1),
    'y': ('pa', 1),
    'j': ('dha', 1),
    'i': ('ni', 1),

    'o': ('sa', 2),
    'p': ('re', 2),
    "'": ('ga', 2),
    ']': ('ma', 2),
    "\\": ('pa', 2),
    '8': ('dha', 2),
    '9': ('ni', 2),
    
    'a': ('ni', 0.5),
    'c': ('dha', 0.5),
    'x': ('pa', 0.5),
    'z': ('ma', 0.5),
    'n': ('sa', 0.5),
    'b': ('re', 0.5),
    'm': ('ga', 0.5)
}

# ---------------- AUDIO ----------------
def make_sound(freq):
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    wave = np.sin(2 * np.pi * freq * t)

    fade = int(0.05 * SAMPLE_RATE)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)

    wave *= env * 0.7

    audio = (wave * 32767).astype(np.int16)
    audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(audio)

# ---------------- ENGINE ----------------
def get_freq(swara, octave):
    rule = RAGA_RULES.get(RAGA, {})

    if swara in rule:
        if rule[swara] == "komal":
            return SA * KOMAL[swara] * octave
        if rule[swara] == "tivra":
            return SA * TIVRA["ma"] * octave

    return SA * SHUDDHA[swara] * octave

# ---------------- STATE ----------------
active_keys = set()
playing = {}
running = True

# ---------------- PRESS ----------------
def on_press(key):
    global running

    try:
        if key == keyboard.Key.esc:
            print("🎹 Exiting...")
            running = False
            return False

        k = key.char

        if k in KEY_MAP and k not in active_keys:
            swara, octave = KEY_MAP[k]

            freq = get_freq(swara, octave)

            sound = make_sound(freq)
            sound.play(-1)

            playing[k] = sound
            active_keys.add(k)

    except:
        pass

# ---------------- RELEASE ----------------
def on_release(key):
    try:
        k = key.char

        if k in active_keys:
            active_keys.remove(k)

        if k in playing:
            playing[k].stop()
            del playing[k]

    except:
        pass

# ---------------- RUN ----------------
print("🎹 Swara Instrument Ready")
print("Press ESC to exit\n")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while running:
    time.sleep(0.01)

listener.stop()
pygame.mixer.quit()



  ============================================================================================================
import pygame
import numpy as np
import time
from pynput import keyboard

# Initialize audio
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

SAMPLE_RATE = 44100
SA = 311.13   # Base Sa (D#)

SHUDDHA = {
    "sa": 1,
    "re": 9/8,
    "ga": 5/4,
    "ma": 4/3,
    "pa": 3/2,
    "dha": 5/3,
    "ni": 15/8
}

KOMAL = {
    "re": 16/15,
    "ga": 6/5,
    "dha": 8/5,
    "ni": 9/5
}

TIVRA = {
    "ma": 45/32
}

KEY_MAP = {
    'w': ('sa', 1),
    'e': ('re', 1),
    'f': ('ga', 1),
    't': ('ma', 1),
    'y': ('pa', 1),
    'j': ('dha', 1),
    'i': ('ni', 1),

    'o': ('sa', 2),
    'p': ('re', 2),
    ';': ('ga', 2),
    '[': ('ma', 2),
    ']': ('pa', 2),

    'a': ('ni', 0.5),
    'c': ('dha', 0.5),
    'x': ('pa', 0.5),
    'z': ('ma', 0.5),
}

playing = {}
press_time = {}
space_pressed = False
running = True


# 🎧 Generate sound
def make_sound(freq):
    t = np.linspace(0, 1, SAMPLE_RATE, False)

    wave = np.sin(2 * np.pi * freq * t)

    wave = wave * 0.6

    audio = (wave * 32767).astype(np.int16)

    # mono → stereo
    audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(audio)


def get_freq(swara, octave):
    if swara == "ma":
        if space_pressed:   # SPACE = Tivra Ma toggle
            ratio = TIVRA["ma"]
            quality = "TIVRA"
        else:
            ratio = SHUDDHA["ma"]
            quality = "SHUDDHA"
    else:
        if space_pressed and swara in KOMAL:
            ratio = KOMAL[swara]
            quality = "KOMAL"
        else:
            ratio = SHUDDHA.get(swara, 1)
            quality = "SHUDDHA"

    return SA * ratio * octave


# 🎹 Key press
def on_press(key):
    global space_pressed, running

    try:
        # ESC to exit
        if key == keyboard.Key.esc:
            print("🎹 Exiting Swara Keyboard...")
            running = False
            return False

        if key == keyboard.Key.space:
            space_pressed = True
            return

        k = key.char

        if k in KEY_MAP and k not in playing:
            swara, octave = KEY_MAP[k]

            press_time[k] = time.time()

            freq = get_freq(swara, octave)
            sound = make_sound(freq)
            sound.play(-1)

            playing[k] = sound

            # determine swara type
            if swara == "ma":
                quality = "TIVRA" if space_pressed else "SHUDDHA"
            else:
                quality = "KOMAL" if (space_pressed and swara in KOMAL) else "SHUDDHA"
                
            # octave label
            if octave == 0.5:
                oct_label = "MANDRA"
            elif octave == 1:
                oct_label = "MADHYA"
            else:
                oct_label = "TAR"

            print(f"🎵 {swara.upper()} | {quality} | {oct_label} | {round(freq,2)} Hz")

    except:
        pass


# 🎹 Key release
def on_release(key):
    global space_pressed

    if key == keyboard.Key.space:
        space_pressed = False
        return

    try:
        k = key.char

        if k in playing:
            playing[k].stop()
            del playing[k]

            if k in press_time:
                duration = time.time() - press_time[k]
                print(f"⏱ Held for {duration:.2f} sec")
                del press_time[k]

    except:
        pass


print("🎶 Swara Keyboard Ready!")
print("➡ Press keys to play notes")
print("➡ Hold SPACE for komal swaras")
print("➡ Press ESC to exit")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while running:
        pass
except KeyboardInterrupt:
    pass

listener.stop()
pygame.mixer.quit()
print("Swara stopped safely")
  
