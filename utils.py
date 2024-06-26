import re
import numpy as np
import librosa


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


"""
This function converts the predictions made by the neural network into a readable format.
"""


def convert_seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 3)
    return f"{hours}h:{minutes}m:{seconds}s"


def preds_to_se(p, audio_clip_length=8.0):
    start_speech = -100
    start_music = -100
    stop_speech = -100
    stop_music = -100

    audio_events = []

    n_frames = p.shape[0]

    if p[0, 0] == 1:
        start_speech = 0

    if p[0, 1] == 1:
        start_music = 0

    for i in range(n_frames - 1):
        if p[i, 0] == 0 and p[i + 1, 0] == 1:
            start_speech = i + 1

        elif p[i, 0] == 1 and p[i + 1, 0] == 0:
            stop_speech = i
            start_time = frames_to_time(start_speech)
            stop_time = frames_to_time(stop_speech)
            audio_events.append((start_time, stop_time, "speech"))
            start_speech = -100
            stop_speech = -100

        if p[i, 1] == 0 and p[i + 1, 1] == 1:
            start_music = i + 1
        elif p[i, 1] == 1 and p[i + 1, 1] == 0:
            stop_music = i
            start_time = frames_to_time(start_music)
            stop_time = frames_to_time(stop_music)
            audio_events.append((start_time, stop_time, "music"))
            start_music = -100
            stop_music = -100

    if start_speech != -100:
        start_time = frames_to_time(start_speech)
        stop_time = audio_clip_length
        audio_events.append((start_time, stop_time, "speech"))
        start_speech = -100
        stop_speech = -100

    if start_music != -100:
        start_time = frames_to_time(start_music)
        stop_time = audio_clip_length
        audio_events.append((start_time, stop_time, "music"))
        start_music = -100
        stop_music = -100

    audio_events.sort(key=lambda x: x[0])
    return audio_events


""" This function was adapted from https://github.com/qlemaire22/speech-music-detection """


def smooth_output(output, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6):
    # This function was adapted from https://github.com/qlemaire22/speech-music-detection
    duration_frame = 220 / 22050
    n_frame = output.shape[1]

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if output[0, i] == 1:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= max_silence_speech:
                    output[0, start_speech:i] = 1

            start_speech = i

        if output[1, i] == 1:
            if i - start_music > 1:
                if (i - start_music) * duration_frame <= max_silence_music:
                    output[1, start_music:i] = 1

            start_music = i

    start_music = -1000
    start_speech = -1000

    for i in range(n_frame):
        if i != n_frame - 1:
            if output[0, i] == 0:
                if i - start_speech > 1:
                    if (i - start_speech) * duration_frame <= min_speech:
                        output[0, start_speech:i] = 0

                start_speech = i

            if output[1, i] == 0:
                if i - start_music > 1:
                    if (i - start_music) * duration_frame <= min_music:
                        output[1, start_music:i] = 0

                start_music = i
        else:
            if i - start_speech > 1:
                if (i - start_speech) * duration_frame <= min_speech:
                    output[0, start_speech:i + 1] = 0

            if i - start_music > 1:
                if (i - start_music) * duration_frame <= min_music:
                    output[1, start_music:i + 1] = 0

    return output


def frames_to_time(f, sr=22050.0, hop_size=220):
    return f * hop_size / sr


def get_log_melspectrogram(audio, sr=22050, hop_length=220, n_fft=1024, n_mels=100, fmin=64, fmax=8000):
    """Return the log-scaled Mel bands of an audio signal."""
    bands = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    return librosa.core.power_to_db(bands, amin=1e-7)
