# Handle colab bug8

from subprocess import Popen, PIPE
import glob
import numpy as np
import random
import soundfile as sf
import librosa
import os


music_files = glob.glob('./musan/music/**/*.wav', recursive = True)
music_files.sort()
speech_files = glob.glob('./musan/speech/**/*.wav', recursive = True)
speech_files.sort()

noise_files = glob.glob('./musan/noise/**/*.wav', recursive = True)
noise_files.sort()

music_files_filt = []
speech_files_filt = []
noise_files_filt = []

min_dur = int(9.1 * 22050)

for m in music_files:
  a, sr = sf.read(m)
  if (a.ndim > 1):
      in_signal_mono = librosa.to_mono(a.T)
      a = np.copy(in_signal_mono)
  if sr != 22050:
      in_signal_22k = librosa.resample(a, orig_sr=sr, target_sr=22050)
      a = np.copy(in_signal_22k)
  if a.shape[0] >= min_dur:
    music_files_filt.append(m)

for s in speech_files:
  a, sr = sf.read(s)
  if (a.ndim > 1):
      in_signal_mono = librosa.to_mono(a.T)
      a = np.copy(in_signal_mono)
  if sr != 22050:
      in_signal_22k = librosa.resample(a, orig_sr=sr, target_sr=22050)
      a = np.copy(in_signal_22k)

  if a.shape[0] >= min_dur:
    speech_files_filt.append(s)

for s in noise_files:
  a, sr = sf.read(s)
  if (a.ndim > 1):
      in_signal_mono = librosa.to_mono(a.T)
      a = np.copy(in_signal_mono)
  if sr != 22050:
      in_signal_22k = librosa.resample(a, orig_sr=sr, target_sr=22050)
      a = np.copy(in_signal_22k)

  if a.shape[0] >= min_dur:
    noise_files_filt.append(s)



music_files_filt.sort()
speech_files_filt.sort()
noise_files_filt.sort()

print(len(music_files_filt))
print(len(speech_files_filt))
print(len(noise_files_filt))

random.seed(4)
random.shuffle(music_files_filt)
random.shuffle(speech_files_filt)
random.shuffle(noise_files_filt)

m_music = len(music_files_filt)
m_speech = len(speech_files_filt)
m_noise = len(noise_files_filt)

split_music = int(0.8 * m_music)
split_speech = int(0.8 * m_speech)
split_noise = int(0.8 * m_noise)

# To synthesise training set
music_list = music_files_filt[0:split_music]
speech_list = speech_files_filt[0:split_speech]
noise_list = noise_files_filt[0:split_noise]

"""
To synthesise validation set, uncomment the below lines
(The original paper used manually annotated radio recordings as the validation set).
"""
# music_list = music_files_filt[split_music:]
# speech_list = speech_files_filt[split_speech:]
# noise_list = noise_files_filt[split_noise:] 

print(len(music_list))
print(len(speech_list))
print(len(noise_list))

random.seed()

"""
This is the mixed (includes music + speech) version of create_transition
"""
def create_mixed_transition(max_f_out_dur = 1.0, max_f_in_dur = 1.0, max_c_fade_dur = 1.0, audio_clip_length = 8.0, min_segment_length = 1.0):
    transition = {}
    transition['type'] = random.choice(["music+speech", "speech_to_music+speech", "music_to_music+speech", "music+speech_to_music", "music+speech_to_speech"])
    if transition['type'] == "speech_to_music+speech":
      transition['music_gain'] = np.random.uniform(0.3, 0.7)
      # transition['music_gain'] is a dummy value. It is set again later according to the loudness normalization.

      transition['f_in_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
      transition['f_in_dur'] = np.random.uniform(0, max_f_in_dur)
      
      if transition['f_in_curve'] == "exp-convex" or transition['f_in_curve'] == "exp-concave" or transition['f_in_curve'] == "s-curve":
        transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.
        
        
    elif transition['type'] == "music_to_music+speech":
      transition['music_gain_1'] = 1.0
      #transition['music_gain_1'] = np.random.uniform(0.7, 1.0)
      transition['music_gain_2'] = np.random.uniform(0.3, 0.7)
      # transition['music_gain_2'] is a dummy value. It is set again later according to the loudness normalization.

      transition['f_in_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
      transition['f_in_dur'] = np.random.uniform(0, max_f_in_dur)
      
      if transition['f_in_curve'] == "exp-convex" or transition['f_in_curve'] == "exp-concave" or transition['f_in_curve'] == "s-curve":
        transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.

      transition['f_out_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
      transition['f_out_dur'] = np.random.uniform(0, max_f_out_dur)
     
      if transition['f_out_curve'] == "exp-convex" or transition['f_out_curve'] == "exp-concave" or transition['f_out_curve'] == "s-curve":
          transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.


    elif transition['type'] == "music+speech_to_music":
      transition['music_gain_1'] = np.random.uniform(0.3, 0.7)
      # transition['music_gain_1'] is a dummy value. It is set again later according to the loudness normalization.
      transition['music_gain_2'] = 1.0

      transition['f_out_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
      transition['f_out_dur'] = np.random.uniform(0, max_f_out_dur)
     
      if transition['f_out_curve'] == "exp-convex" or transition['f_out_curve'] == "exp-concave" or transition['f_out_curve'] == "s-curve":
          transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.

      transition['f_in_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
      transition['f_in_dur'] = np.random.uniform(0, max_f_in_dur)
      
      if transition['f_in_curve'] == "exp-convex" or transition['f_in_curve'] == "exp-concave" or transition['f_in_curve'] == "s-curve":
        transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.

    elif transition['type'] == "music+speech_to_speech":
      transition['music_gain'] = np.random.uniform(0.3, 0.7)

      transition['f_out_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
      transition['f_out_dur'] = np.random.uniform(0, max_f_out_dur)

      if transition['f_out_curve'] == "exp-convex" or transition['f_out_curve'] == "exp-concave" or transition['f_out_curve'] == "s-curve":
          transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.

    elif transition['type'] == "music+speech":
      transition['music_gain'] = np.random.uniform(0.3, 0.7)
      # transition['music_gain'] is a dummy value. It is set again later according to the loudness normalization.

    # ======= Calculate the time of transition in the same function ========
    if transition['type'] == "music+speech":
      return (transition, -1.0)
    
    else:
      point = np.random.uniform(min_segment_length + max_f_out_dur, audio_clip_length - min_segment_length - max_f_in_dur)
      return (transition, point)

def create_mixed_samples_list(music_sounds, speech_sounds):
    """
    Returns a dictionary containing music and speech sounds.
    Take the class_list as input and randomly pick sound files in the `music_sounds` and `speech_sounds` folder.
    """
    samples = {}

    cc = random.choice(music_sounds)
    samples['music'] = cc

    cc = random.choice(speech_sounds)
    samples['speech'] = cc

    return samples

def get_mixed_segment_lengths(transition, audio_clip_length=8.0, sr = 22050):
  """
  This function returns a dictionary.
  """
  segment_lengths = {}

  ac_len_samples = int(audio_clip_length * sr)
  t_samples = int(transition[1] * sr) # Transition time in samples

  if transition[0]['type'] == "music+speech":
    segment_lengths['music'] = ac_len_samples
    segment_lengths['speech'] = ac_len_samples

  elif transition[0]['type'] == "speech_to_music+speech":
    segment_lengths['speech'] = ac_len_samples
    segment_lengths['music'] = ac_len_samples - t_samples

  elif transition[0]['type'] == "music_to_music+speech":
    segment_lengths['speech'] = ac_len_samples - t_samples
    segment_lengths['music'] = ac_len_samples

  elif transition[0]['type'] == "music+speech_to_music":
    segment_lengths['music'] = ac_len_samples
    segment_lengths['speech'] = t_samples

  elif transition[0]['type'] == "music+speech_to_speech":
    segment_lengths['speech'] = ac_len_samples
    segment_lengths['music'] = t_samples

  return segment_lengths

def get_mixed_random_segments(samples, segment_lengths, f_buffer = 0.0, sr = 22050):
    """
    This function returns a dictionary of tuples that specifies the segment boundaries in the original sound file.
    """   

    f_buffer_samples = int(f_buffer * sr)

    segments = {}     
    d, sr = sf.read(samples['speech'])
    if (d.ndim > 1):
        in_signal_mono = librosa.to_mono(d.T)
        d = np.copy(in_signal_mono)
    if sr != 22050:
        in_signal_22k = librosa.resample(d, orig_sr=sr, target_sr=22050)
        d = np.copy(in_signal_22k)

    sample_length = len(d)
    r = np.random.randint(f_buffer_samples, sample_length - segment_lengths['speech'] - f_buffer_samples)
    segments['speech'] = (r, r + segment_lengths['speech'])
    
    d, sr = sf.read(samples['music'])
    if (d.ndim > 1):
        in_signal_mono = librosa.to_mono(d.T)
        d = np.copy(in_signal_mono)
    if sr != 22050:
        in_signal_22k = librosa.resample(d, orig_sr=sr, target_sr=22050)
        d = np.copy(in_signal_22k)
    sample_length = len(d)
    r = np.random.randint(f_buffer_samples, sample_length - segment_lengths['music'] - f_buffer_samples)
    segments['music'] = (r, r + segment_lengths['music'])

    return segments

def apply_mixed_fade_out(audio, transition, sr=22050.0, end_gain = 0.0):
  stop = audio.shape[0]
  f_out_length_samples =  int(transition[0]['f_out_dur'] * sr)

  if transition[0]['f_out_curve'] == "linear":     
      audio[stop - f_out_length_samples:stop] = audio[stop - f_out_length_samples:stop] * np.linspace(1.0, end_gain, num = f_out_length_samples)

  elif transition[0]['f_out_curve'] == "exp-concave":
      a = np.linspace(1.0, 0.0, num = f_out_length_samples)
      x = transition[0]['exp_value']
      fade_curve = a ** x
      fade_curve = fade_curve * (1 - end_gain) + end_gain
      audio[stop - f_out_length_samples:stop] = audio[stop - f_out_length_samples:stop] * fade_curve
      
  elif transition[0]['f_out_curve'] == "exp-convex":
      a = np.linspace(0.0, 1.0, num = f_out_length_samples)
      x = transition[0]['exp_value']
      fade_curve = 1 - a ** x
      fade_curve = fade_curve * (1 - end_gain) + end_gain
      audio[stop - f_out_length_samples:stop] = audio[stop - f_out_length_samples:stop] * fade_curve
      
  elif transition[0]['f_out_curve'] == "s-curve":
      n_1 = int(f_out_length_samples / 2)
      a_1 = np.linspace(0.0, 1.0, num = n_1)
      a_2 = np.linspace(0.0, 1.0, num = f_out_length_samples - n_1)
      x = transition[0]['exp_value']
      
      convex = 0.5 * (1 - a_1 ** x) + 0.5
      
      concave = 0.5 * (1 - a_2)  ** x
      
      fade_curve = np.concatenate((convex, concave))
      fade_curve = fade_curve * (1 - end_gain) + end_gain
      
      audio[stop - f_out_length_samples:stop] = audio[stop - f_out_length_samples:stop] * fade_curve

def apply_mixed_normal_fade_in(audio, transition, sr=22050.0, end_gain = 1.0, start_gain = 0.0):
  start = 0
  f_in_length_samples =  int(transition[0]['f_in_dur'] * sr)  

  #print('f_in_length_samples is {}'.format(f_in_length_samples))

  #print("audio.shape is {}".format(audio.shape))

  if transition[0]['f_in_curve'] == "linear":        
    audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * np.linspace(start_gain, end_gain, num = f_in_length_samples)      

  elif transition[0]['f_in_curve'] == "exp-concave":
    a = np.linspace(0.0, 1.0, num = f_in_length_samples)
    x = transition[0]['exp_value']
    fade_curve = a ** x
    fade_curve = fade_curve * (end_gain - start_gain) + start_gain
    audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * fade_curve
      
  elif transition[0]['f_in_curve'] == "exp-convex":
    a = np.linspace(1.0, 0.0, num = f_in_length_samples)
    x = transition[0]['exp_value']
    fade_curve = 1 - a ** x
    fade_curve = fade_curve * (end_gain - start_gain) + start_gain
    audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * fade_curve
      
  elif transition[0]['f_in_curve'] == "s-curve":
    n_1 = int(f_in_length_samples / 2)
    a_1 = np.linspace(0.0, 1.0, num = n_1)
    a_2 = np.linspace(0.0, 1.0, num = f_in_length_samples - n_1)
    x = transition[0]['exp_value']
    
    concave = 0.5 * a_1 ** x
    
    convex = 0.5 * (1 - (1 - a_2)  ** x) + 0.5
    
    fade_curve = np.concatenate((concave, convex))
    fade_curve = fade_curve * (end_gain - start_gain) + start_gain
    audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * fade_curve

def generate_mixed_multiclass_labels(transition, audio_clip_length = 8.0, sr = 22050.0, res = 220):
  """
  This function generates multiclass labels for music+speech examples.
  `res` is in samples.
  """
  res_t = 220 / sr
  no_of_labels = int(np.ceil(audio_clip_length / res_t))
  t_point = int(transition[1] / res_t)
  
  if 'f_out_dur' in transition[0]:
    f_out_samples = int(transition[0]['f_out_dur'] / res_t)
  
  if 'f_in_dur' in transition[0]:
    f_in_samples = int(transition[0]['f_in_dur'] / res_t)

  labels = np.zeros((no_of_labels, 2), dtype = np.int16)

  """
  "music+speech", "speech_to_music+speech", "music_to_music+speech", "music+speech_to_music", "music+speech_to_speech"
  """

  if transition[0]['type'] == "music+speech":
    labels[:, 0] = 1
    labels[:, 1] = 1

  elif transition[0]['type'] == "speech_to_music+speech":
    labels[:, 0] = 1
    labels[t_point:, 1] = 1

  elif transition[0]['type'] == "music_to_music+speech":
    labels[t_point:, 0] = 1
    labels[:, 1] = 1

  elif transition[0]['type'] == "music+speech_to_music":
    labels[0:t_point, 0] = 1
    labels[:, 1] = 1

  elif transition[0]['type'] == "music+speech_to_speech":
    labels[:, 0] = 1
    labels[0:t_point, 1] = 1
  
  return labels

import pyloudnorm as pyln
def get_random_loudness_gain(speech_data, music_data, rate = 22050):
  meter = pyln.Meter(rate)
  speech_loudness = meter.integrated_loudness(speech_data)
  music_loudness = meter.integrated_loudness(music_data)
  random_loudness = np.random.uniform(speech_loudness - 18.0, speech_loudness - 7.0)
  delta_loudness = random_loudness - music_loudness
  gain = np.power(10.0, delta_loudness/20.0)

  #print("The gain is {}".format(gain))

  return gain

"""
This would create an audio clip template based on the output of create_mixed_transition.
"""
"""
"music+speech", "speech_to_music+speech", "music_to_music+speech", "music+speech_to_music", "music+speech_to_speech"
"""

def create_mixed_audio_clip(audio_clip_length = 8.0, sr = 22050.0):
  transition = create_mixed_transition()
  #print(transition)
  samples = create_mixed_samples_list(music_sounds, speech_sounds)
  segment_lengths = get_mixed_segment_lengths(transition)
  segments = get_mixed_random_segments(samples, segment_lengths)

  #print(segments)

  start_sp = segments['speech'][0]
  stop_sp = segments['speech'][1]

  start_mu = segments['music'][0]
  stop_mu = segments['music'][1]

  if transition[0]['type'] == "music+speech":
    synth_audio, _ = sf.read(samples['speech'], start = start_sp, stop = stop_sp)
    if (synth_audio.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_audio.T)
        synth_audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_audio, orig_sr=_, target_sr=22050)
        synth_audio = np.copy(in_signal_22k)
    synth_audio = librosa.util.normalize(synth_audio)

    synth_music, _ = sf.read(samples['music'], start = start_mu, stop = stop_mu)
    if (synth_music.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_music.T)
        synth_music = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_music, orig_sr=_, target_sr=22050)
        synth_music = np.copy(in_signal_22k)

    synth_music = librosa.util.normalize(synth_music)

    m_gain = get_random_loudness_gain(synth_audio, synth_music)

    synth_audio += m_gain * synth_music

  elif transition[0]['type'] == "speech_to_music+speech":
    synth_audio, _ = sf.read(samples['speech'], start = start_sp, stop = stop_sp)
    if (synth_audio.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_audio.T)
        synth_audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_audio, orig_sr=_, target_sr=22050)
        synth_audio = np.copy(in_signal_22k)

    synth_audio = librosa.util.normalize(synth_audio)

    music_start_point = int(transition[1] * sr)

    synth_music, _ = sf.read(samples['music'], start = start_mu, stop = stop_mu)
    if (synth_music.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_music.T)
        synth_music = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_music, orig_sr=_, target_sr=22050)
        synth_music = np.copy(in_signal_22k)

    synth_music = librosa.util.normalize(synth_music)

    m_gain = get_random_loudness_gain(synth_audio, synth_music)

    apply_mixed_normal_fade_in(synth_music, transition, sr=22050.0, end_gain = m_gain)

    f_in_length_samples =  int(transition[0]['f_in_dur'] * sr)  

    synth_music[f_in_length_samples:] = synth_music[f_in_length_samples:] * m_gain

    synth_audio[music_start_point:] += synth_music

  elif transition[0]['type'] == "music_to_music+speech":
    synth_audio, _ = sf.read(samples['music'], start = start_mu, stop = stop_mu)
    if (synth_audio.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_audio.T)
        synth_audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_audio, orig_sr=_, target_sr=22050)
        synth_audio = np.copy(in_signal_22k)

    synth_audio = librosa.util.normalize(synth_audio)
    
    music1_end_point = int(transition[1] * sr)
    synth_music1 = synth_audio[0:music1_end_point]
    synth_music1 = synth_music1 * transition[0]['music_gain_1']
    #print('synth_music1.shape is {}'.format(synth_music1.shape))

    synth_speech, _ = sf.read(samples['speech'], start = start_sp, stop = stop_sp)
    if (synth_speech.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_speech.T)
        synth_speech = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_speech, orig_sr=_, target_sr=22050)
        synth_speech = np.copy(in_signal_22k)

    synth_speech = librosa.util.normalize(synth_speech)

    apply_mixed_normal_fade_in(synth_speech, transition, sr=22050.0)

    m_gain = get_random_loudness_gain(synth_speech, synth_audio)
    apply_mixed_fade_out(synth_music1, transition, sr=22050.0, end_gain = m_gain)

    synth_music2 = synth_audio[music1_end_point:]
    synth_music2 = synth_music2 * m_gain

    synth_audio[0:music1_end_point] = synth_music1
    synth_audio[music1_end_point:] = synth_speech + synth_music2

  elif transition[0]['type'] == "music+speech_to_music":
    synth_audio, _ = sf.read(samples['music'], start = start_mu, stop = stop_mu)
    if (synth_audio.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_audio.T)
        synth_audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_audio, orig_sr=_, target_sr=22050)
        synth_audio = np.copy(in_signal_22k)

    synth_audio = librosa.util.normalize(synth_audio)

    speech_start_point = int(transition[1] * sr)

    synth_speech, _ = sf.read(samples['speech'], start = start_sp, stop = stop_sp)
    if (synth_speech.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_speech.T)
        synth_speech = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_speech, orig_sr=_, target_sr=22050)
        synth_speech = np.copy(in_signal_22k)
    synth_speech = librosa.util.normalize(synth_speech)

    apply_mixed_fade_out(synth_speech, transition, sr=22050.0)

    music1_end_point = int(transition[1] * sr)
    synth_music1 = synth_audio[0:music1_end_point]

    m_gain = get_random_loudness_gain(synth_speech, synth_audio)

    synth_music1 = synth_music1 * m_gain

    synth_music2 = synth_audio[music1_end_point:]
    f_in_length_samples =  int(transition[0]['f_in_dur'] * sr)
    apply_mixed_normal_fade_in(synth_music2, transition, sr=22050.0, start_gain = m_gain, end_gain = transition[0]['music_gain_2'])
    synth_music2[f_in_length_samples:] = synth_music2[f_in_length_samples:] * transition[0]['music_gain_2']

    synth_audio[0:music1_end_point] = synth_speech + synth_music1

    synth_audio[music1_end_point:] = synth_music2

  elif transition[0]['type'] == "music+speech_to_speech":
    synth_audio, _ = sf.read(samples['speech'], start = start_sp, stop = stop_sp)
    if (synth_audio.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_audio.T)
        synth_audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_audio, orig_sr=_, target_sr=22050)
        synth_audio = np.copy(in_signal_22k)

    synth_audio = librosa.util.normalize(synth_audio)

    music_end_point = int(transition[1] * sr)

    synth_music, _ = sf.read(samples['music'], start = start_mu, stop = stop_mu)
    if (synth_music.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_music.T)
        synth_music = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_music, orig_sr=_, target_sr=22050)
        synth_music = np.copy(in_signal_22k)


    synth_music = librosa.util.normalize(synth_music)

    m_gain = get_random_loudness_gain(synth_audio, synth_music)

    synth_music = synth_music * m_gain
    apply_mixed_fade_out(synth_music, transition, sr=22050.0)

    synth_audio[0:music_end_point] += synth_music

  return (synth_audio, transition)

def get_log_melspectrogram(audio, sr = 22050, hop_length = 220, n_fft = 1024, n_mels = 100, fmin = 64, fmax = 8000):
    """Return the log-scaled Mel bands of an audio signal."""
    # default 
    # sr = 22050, hop_length = 220, n_fft = 1024, n_mels = 80, fmin = 64, fmax = 8000
    bands = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    return librosa.core.power_to_db(bands, amin=1e-7)

"""
This function is used to divide the synthesised into different folders called blocks.
"""

def get_block_id(mel_id, block_size = 320):
  i = int(mel_id.replace("mel-id-", ""))
  b = int((i - 1) // block_size)
  return b + 1

def check_overlap(transition_points, point, min_segment_length):
    is_overlap = False
    for t in transition_points:
        if np.absolute(point - t) <= min_segment_length + 2.0: # I am adding 4.0 to separate out the files.
            is_overlap = True
    return is_overlap

def create_random_transition_points(audio_clip_length, min_segment_length = 1.0, max_f_out_dur = 0.5, max_f_in_dur = 0.0):   
    # If max_no_transitions = 2, then the audio example can have maximum of 1 transition (ie., max_no_transitions - 1)
    max_no_transitions = 2
    number_of_transitions = np.random.randint(0, max_no_transitions)
    #print("Number of transitions is {}".format(number_of_transitions))
    if number_of_transitions == 0:
        return []
    transition_points = [np.random.uniform(min_segment_length + max_f_out_dur, audio_clip_length - min_segment_length - max_f_in_dur)]
    #print(transition_points)
    
    # Limit number of iterations 
    num_iters = 100000
    
    while len(transition_points) < number_of_transitions:
        point = np.random.uniform(min_segment_length + max_f_out_dur, audio_clip_length - min_segment_length - max_f_in_dur)
        if not check_overlap(transition_points, point, min_segment_length):
            transition_points.append(point)
            num_iters = 100000
        else: 
            num_iters -= 1
            if num_iters < 0:
                #print('Unable to find the required number of transition points. The minimum segment length seems to be high!!')
                
                # Re-calculate the number of transitions and the first transition_point.
                number_of_transitions = np.random.randint(0, max_no_transitions)
                #print("Number of transitions is {}".format(number_of_transitions))
                if number_of_transitions == 0:
                    return []
                transition_points = [np.random.uniform(min_segment_length + max_f_out_dur, audio_clip_length - min_segment_length - max_f_in_dur)]
                #print(transition_points)
                continue                
                raise ValueError('Unable to find the required number of transition points. The minimum segment length seems to be high!!')
    transition_points.sort()
    return transition_points

def create_transition(max_f_out_dur = 1.0, max_f_in_dur = 1.0, max_c_fade_dur = 1.0, max_time_gap = 0.2):
    """
    Returns a dictionary containing parameters of the transition.
    For a normal fade, it is the following.
    {type, f_out_curve, f_out_dur, time_gap, f_in_curve, f_in_dur}.
    For a cross-fade it is the following.
    {type, f_out_curve, f_out_dur, f_in_curve, f_in_dur}.
    """
    transition = {}
    transition['type'] = random.choice(['normal', 'cross-fade'])
    if transition['type'] == "normal":
        transition['f_out_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
        transition['f_out_dur'] = np.random.uniform(0, max_f_out_dur)
        transition['time_gap'] = np.random.uniform(0.0, max_time_gap) # I am setting this to only positive values for the moment.
        transition['f_in_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
        transition['f_in_dur'] = np.random.uniform(0, max_f_in_dur)
        
        if transition['f_out_curve'] == "exp-convex" or transition['f_out_curve'] == "exp-concave" or transition['f_out_curve'] == "s-curve":
            transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.

        if transition['f_in_curve'] == "exp-convex" or transition['f_in_curve'] == "exp-concave" or transition['f_in_curve'] == "s-curve":
            transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.
        
        
    elif transition['type'] == "cross-fade":
        transition['f_out_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
        transition['f_out_dur'] = np.random.uniform(0, max_c_fade_dur)
        transition['f_in_curve'] = random.choice(['linear', 'exp-convex', 'exp-concave', 's-curve'])
        transition['f_in_dur'] = np.random.uniform(0, max_c_fade_dur)        

        if transition['f_out_curve'] == "exp-convex" or transition['f_out_curve'] == "exp-concave" or transition['f_out_curve'] == "s-curve":
            transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp and exp-convex transitions.

        if transition['f_in_curve'] == "exp-convex" or transition['f_in_curve'] == "exp-concave" or transition['f_in_curve'] == "s-curve":
            transition['exp_value'] = np.random.uniform(1.5, 3.0) # This is the additional `exp_value` that is calculated only for exp-concave and exp-convex transitions.
    
    #print(transition)
    return transition

def create_transition_list(transition_points):
    """
    This function returns a list of transitions. 
    Each element in the list is tuple (transition, time_stamp)
    """
    transitions_list = []
    for i in range(len(transition_points)):
      if len(transition_points) == 1 or i == len(transition_points) - 1:
        t = create_transition(max_time_gap = transition_points[i] - 1.0)
        transitions_list.append((t, transition_points[i]))

      elif i == 0:
        #print("Central section reached!!!")
        s_len = transition_points[i]
        #print("s_len is {}".format(s_len))
        t = create_transition(max_f_out_dur=min(0.2 * s_len, 1.0), max_f_in_dur=min(0.1 * s_len, 1.0), max_c_fade_dur=min(0.1 * s_len, 1.0))
        transitions_list.append((t, transition_points[i]))        
      else:
        #print("Middle section reached!!!")
        s_len = transition_points[i] - transition_points[i - 1]
        #print("s_len is {}".format(s_len))
        t = create_transition(max_f_out_dur=min(0.2 * s_len, 1.0), max_f_in_dur=min(0.1 * s_len, 1.0), max_c_fade_dur=min(0.1 * s_len, 1.0), max_time_gap = transition_points[i] - 1.0)
        transitions_list.append((t, transition_points[i]))

    return transitions_list

def create_class_list(no_of_classes):
    """
    Create a random list of classes.
    """
    # Create a random list containing music or speech.
    class_list = random.choices(['music', 'speech', 'noise'], weights=[0.4, 0.4, 0.2], k=no_of_classes)
    # class_list = random.choices(['music', 'speech', 'noise'], weights=[0.47, 0.47, 0.06], k=no_of_classes)
    return class_list

def create_samples_list(class_list, music_sounds, speech_sounds, noise_sounds):
    """
    Take the class_list as input and randomly pick sound files in the `music_sounds` and `speech_sounds` folder.
    """
    samples_list = []
    for c in class_list:
        if (c == "music"):
            cc = random.choice(music_sounds)
            samples_list.append(cc)
        elif (c == "speech"):
            cc = random.choice(speech_sounds)
            samples_list.append(cc)
        elif (c == "noise"):
            cc = random.choice(noise_sounds)
            samples_list.append(cc)

        else:
            print("Encountered unexpected class!!")
            raise ValueError("Encountered unexpected class!!")
    return samples_list

def get_segment_lengths(transitions_list, audio_clip_length):
    """
    This function takes the list of transitions as input.
    It returns the length of each segment.
    """
    segment_lengths = []
    # Extract the time_stamps from transitions_list
    time_stamps = [j for (i, j) in transitions_list]
    time_stamps = [0] + time_stamps + [audio_clip_length]
    for t in range(len(time_stamps) - 1):
        tt = time_stamps[t + 1] - time_stamps[t]
        segment_lengths.append(tt)    
    return segment_lengths

def get_random_segments(samples_list, segment_lengths, f_buffer = 1.1):
    """
    This function picks random segments from the samples_list. 
    It returns a list of tuples (segment_start, segment_end)
    """
    if len(samples_list) != len(segment_lengths):
        print("The length of samples_list needs to be equal to segment_lengths!!")
        raise ValueError("Data mismatch --- The length of samples_list needs to be equal to segment_lengths!!")
    
    segments = []
    
    for i in range(len(samples_list)):
        #remove_silence_and_resample(samples_list[i])
        
        d, sr = sf.read(samples_list[i])
        if (d.ndim > 1):
            in_signal_mono = librosa.to_mono(d.T)
            d = np.copy(in_signal_mono)
        if sr != 22050:
            in_signal_22k = librosa.resample(d, orig_sr=sr, target_sr=22050)
            d = np.copy(in_signal_22k)
        sample_length = float(len(d) / sr)
        r = np.random.uniform(f_buffer, sample_length - segment_lengths[i] - f_buffer)
        segments.append((r, r + segment_lengths[i]))
    return segments

def create_template_audio_clip(audio_clip_length, samples_list, segments, sr):
    """
    This stitches all the individual audio segments into one file. It does not include the transitions.
    It returns `synth_audio` which is the synthesised audio file.
    It also returns `synth_audio_seg_samples`, which is a list of tuples (audio clip start, audio clip stop).
    These tuples serve as reference points to perform fade in and fade out operations.
    """
    if (len(samples_list) < 1):
        print("The samples_list argument is invalid!!")
        raise ValueError("The samples_list argument is invalid!!")
    
    synth_audio_seg_samples = [] # This is a list of tuples containing segment boundaries in the synthesised audio.
    
    #print("segments[0]: {}".format(segments[0]))
    
    start = int(segments[0][0] * sr)
    stop = int(np.ceil(segments[0][1] * sr))
    ac_start = 0 # Synthesised audio clip start
    ac_stop = stop - start
    synth_audio_seg_samples.append((ac_start, ac_stop))
    synth_audio, _ = sf.read(samples_list[0], start = start, stop = stop)
    if (synth_audio.ndim > 1):
        in_signal_mono = librosa.to_mono(synth_audio.T)
        synth_audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(synth_audio, orig_sr=_, target_sr=22050)
        synth_audio = np.copy(in_signal_22k)

    synth_audio = librosa.util.normalize(synth_audio)

    for i in range(1, len(samples_list)):

        start = int(segments[i][0] * sr)
        stop = int(np.ceil(segments[i][1] * sr))
        ac_start = ac_stop # I have just removed the `+ 1` from the equation.
        ac_stop = ac_start + stop - start
        synth_audio_seg_samples.append((ac_start, ac_stop))
        sa, _ = sf.read(samples_list[i], start = start, stop = stop)
        if (sa.ndim > 1):
            in_signal_mono = librosa.to_mono(sa.T)
            sa = np.copy(in_signal_mono)
        if _ != 22050:
            in_signal_22k = librosa.resample(sa, orig_sr=_, target_sr=22050)
            sa = np.copy(in_signal_22k)

        sa = librosa.util.normalize(sa)
        synth_audio = np.concatenate((synth_audio, sa), axis = 0)
    
    return synth_audio, synth_audio_seg_samples
def apply_normal_fade_out(audio, transition, synth_audio_seg_samples, sr):
    """
    This function applies the fade out operation on the `audio` array directly.
    `synth_audio_seg_samples` is a tuple (start, stop), that contains the reference points to perform
    fade out and fade in operations.
    """       
    start, stop = synth_audio_seg_samples
    
    f_out_length_samples =  int(transition[0]['f_out_dur'] * sr)
    
    
    # `stop_shrunk` refers to the new end point after silencing `time_gap` samples. 
    stop_shrunk = stop - int(transition[0]['time_gap'] * sr)
    
    # Set all the samples in the time gap to be 0.
    audio[stop_shrunk:stop] = 0.0     
    
    #print("stop_shrunk: {}".format(stop_shrunk))
    
    if transition[0]['f_out_curve'] == "linear":     
        audio[stop_shrunk - f_out_length_samples:stop_shrunk] = audio[stop_shrunk - f_out_length_samples:stop_shrunk] * np.linspace(1.0, 0.0, num = f_out_length_samples)

    elif transition[0]['f_out_curve'] == "exp-concave":
        a = np.linspace(1.0, 0.0, num = f_out_length_samples)
        x = transition[0]['exp_value']
        fade_curve = a ** x
        audio[stop_shrunk - f_out_length_samples:stop_shrunk] = audio[stop_shrunk - f_out_length_samples:stop_shrunk] * fade_curve
        
    elif transition[0]['f_out_curve'] == "exp-convex":
        a = np.linspace(0.0, 1.0, num = f_out_length_samples)
        x = transition[0]['exp_value']
        fade_curve = 1 - a ** x
        audio[stop_shrunk - f_out_length_samples:stop_shrunk] = audio[stop_shrunk - f_out_length_samples:stop_shrunk] * fade_curve
       
    elif transition[0]['f_out_curve'] == "s-curve":
        n_1 = int(f_out_length_samples / 2)
        a_1 = np.linspace(0, 1, num = n_1)
        a_2 = np.linspace(0, 1, num = f_out_length_samples - n_1)
        x = transition[0]['exp_value']
        
        convex = 0.5 * (1 - a_1 ** x) + 0.5
        
        concave = 0.5 * (1 - a_2)  ** x
        
        fade_curve = np.concatenate((convex, concave))
        
        audio[stop_shrunk - f_out_length_samples:stop_shrunk] = audio[stop_shrunk - f_out_length_samples:stop_shrunk] * fade_curve

def apply_normal_fade_in(audio, transition, synth_audio_seg_samples, sr):
    """
    This function applies the fade in operation on the `audio` array directly.
    `synth_audio_seg_samples` is a tuple (start, stop), that contains the reference points to perform
    fade out and fade in operations.    
    """
    start, stop = synth_audio_seg_samples
    f_in_length_samples =  int(transition[0]['f_in_dur'] * sr)
    
    if transition[0]['f_in_curve'] == "linear":        
        audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * np.linspace(0.0, 1.0, num = f_in_length_samples)      

    elif transition[0]['f_in_curve'] == "exp-concave":
        a = np.linspace(0.0, 1.0, num = f_in_length_samples)
        x = transition[0]['exp_value']
        fade_curve = a ** x
        audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * fade_curve
        
    elif transition[0]['f_in_curve'] == "exp-convex":
        a = np.linspace(1.0, 0.0, num = f_in_length_samples)
        x = transition[0]['exp_value']
        fade_curve = 1 - a ** x
        audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * fade_curve
        
    elif transition[0]['f_in_curve'] == "s-curve":
        n_1 = int(f_in_length_samples / 2)
        a_1 = np.linspace(0, 1, num = n_1)
        a_2 = np.linspace(0, 1, num = f_in_length_samples - n_1)
        x = transition[0]['exp_value']
        
        concave = 0.5 * a_1 ** x
        
        convex = 0.5 * (1 - (1 - a_2)  ** x) + 0.5
        
        fade_curve = np.concatenate((concave, convex))
        
        audio[start:start + f_in_length_samples] = audio[start:start + f_in_length_samples] * fade_curve     

def apply_cross_fade_out(audio, transition, sample, segment, synth_audio_seg_samples, sr):
    """
    This function applies the fade out portion of the cross-fade operation on the `audio` array directly.
    `sample` refers to the sample that is going to fade out.
    `segments` refers to the segment boundaries in the original sound sample.
    `synth_audio_seg_samples` is a tuple (start, stop), that contains the reference points 
    in the synthesised audio clip to perform fade out and fade in operations.      
    """
    f_out_dur_samples = int(transition[0]['f_out_dur'] * sr)

    if f_out_dur_samples > 0:
      
      start, stop = segment
      start_sample = int(start * sr)
      stop_sample = int(stop * sr)
      
      synth_audio_start, synth_audio_stop = synth_audio_seg_samples
      
      cf_out_audio, _ = sf.read(sample, start = stop_sample, stop = stop_sample + f_out_dur_samples)
      if (cf_out_audio.ndim > 1):
          in_signal_mono = librosa.to_mono(cf_out_audio.T)
          cf_out_audio = np.copy(in_signal_mono)
      if _ != 22050:
          in_signal_22k = librosa.resample(cf_out_audio, orig_sr=_, target_sr=22050)
          cf_out_audio = np.copy(in_signal_22k)


      cf_out_audio = librosa.util.normalize(cf_out_audio)
      
      
      if transition[0]['f_out_curve'] == "linear":   
          cf_out_audio = cf_out_audio * np.linspace(1.0, 0.0, num = f_out_dur_samples)
          audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] = audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] + cf_out_audio

      elif transition[0]['f_out_curve'] == "exp-concave":
          a = np.linspace(1.0, 0.0, num = f_out_dur_samples)
          x = transition[0]['exp_value']
          fade_curve = a ** x
          cf_out_audio = cf_out_audio * fade_curve
          audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] = audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] + cf_out_audio
          
      elif transition[0]['f_out_curve'] == "exp-convex":
          a = np.linspace(0.0, 1.0, num = f_out_dur_samples)
          x = transition[0]['exp_value']
          fade_curve = 1 - a ** x        
          cf_out_audio = cf_out_audio * fade_curve
          audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] = audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] + cf_out_audio

      elif transition[0]['f_out_curve'] == "s-curve":
          n_1 = int(f_out_dur_samples / 2)
          a_1 = np.linspace(0, 1, num = n_1)
          a_2 = np.linspace(0, 1, num = f_out_dur_samples - n_1)
          x = transition[0]['exp_value']
          
          convex = 0.5 * (1 - a_1 ** x) + 0.5
          
          concave = 0.5 * (1 - a_2)  ** x
          
          fade_curve = np.concatenate((convex, concave))
          
          cf_out_audio = cf_out_audio * fade_curve
          
          audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] = audio[synth_audio_stop:synth_audio_stop + f_out_dur_samples] + cf_out_audio

def apply_cross_fade_in(audio, transition, sample, segment, synth_audio_seg_samples, sr):
    """
    This function applies the fade in portion of the cross-fade operation on the `audio` array directly.
    `sample` refers to the sample that is going to fade in.
    `segments` refers to the segment boundaries in the original sound sample.
    `synth_audio_seg_samples` is a tuple (start, stop), that contains the reference points 
    in the synthesised audio clip to perform fade out and fade in operations.      
    """    
    f_in_dur_samples = int(transition[0]['f_in_dur'] * sr)

    if f_in_dur_samples > 0:
    
      start, stop = segment
      start_sample = int(start * sr)
      stop_sample = int(stop * sr)    
      
      synth_audio_start, synth_audio_stop = synth_audio_seg_samples
      
      cf_out_audio, _ = sf.read(sample,start = start_sample - f_in_dur_samples, stop = start_sample)
      if (cf_out_audio.ndim > 1):
          in_signal_mono = librosa.to_mono(cf_out_audio.T)
          cf_out_audio = np.copy(in_signal_mono)
      if _ != 22050:
          in_signal_22k = librosa.resample(cf_out_audio, orig_sr=_, target_sr=22050)
          cf_out_audio = np.copy(in_signal_22k)


      cf_out_audio = librosa.util.normalize(cf_out_audio)
      

      if transition[0]['f_in_curve'] == "linear":   
          cf_out_audio = cf_out_audio * np.linspace(0.0, 1.0, num = f_in_dur_samples)    
          audio[synth_audio_start - f_in_dur_samples:synth_audio_start] = audio[synth_audio_start - f_in_dur_samples:synth_audio_start] + cf_out_audio

      elif transition[0]['f_in_curve'] == "exp-concave":
          a = np.linspace(0.0, 1.0, num = f_in_dur_samples)
          x = transition[0]['exp_value']
          fade_curve = a ** x    
          cf_out_audio = cf_out_audio * fade_curve    
          audio[synth_audio_start - f_in_dur_samples:synth_audio_start] = audio[synth_audio_start - f_in_dur_samples:synth_audio_start] + cf_out_audio
          
      elif transition[0]['f_in_curve'] == "exp-convex":
          a = np.linspace(1.0, 0.0, num = f_in_dur_samples)
          x = transition[0]['exp_value']
          fade_curve = 1 - a ** x  
          cf_out_audio = cf_out_audio * fade_curve 
          audio[synth_audio_start - f_in_dur_samples:synth_audio_start] = audio[synth_audio_start - f_in_dur_samples:synth_audio_start] + cf_out_audio

      elif transition[0]['f_in_curve'] == "s-curve":
          n_1 = int(f_in_dur_samples / 2)
          a_1 = np.linspace(0, 1, num = n_1)
          a_2 = np.linspace(0, 1, num = f_in_dur_samples - n_1)
          x = transition[0]['exp_value']
          
          concave = 0.5 * a_1 ** x
          
          convex = 0.5 * (1 - (1 - a_2)  ** x) + 0.5
          
          fade_curve = np.concatenate((concave, convex))
          
          cf_out_audio = cf_out_audio * fade_curve 
          
          audio[synth_audio_start - f_in_dur_samples:synth_audio_start] = audio[synth_audio_start - f_in_dur_samples:synth_audio_start] + cf_out_audio      

def create_audio_clip(audio_clip_length, transitions_list, samples_list, segments, sr):
    """
    This function returns the synthesised audio clip after applying transitions.
    """
    if (len(samples_list) < 1):
        print("The samples_list argument is invalid!!")
        raise ValueError("The samples_list argument is invalid!!")
    
    # Add the first audio segment.    
    a, _ = sf.read(samples_list[0])
    if (a.ndim > 1):
        in_signal_mono = librosa.to_mono(a.T)
        a = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(a, orig_sr=_, target_sr=22050)
        a = np.copy(in_signal_22k)


    a = librosa.util.normalize(a)
    
    synth_audio = np.array([], dtype = np.float32)
    # Add the first transition
    if len(transitions_list) == 0:
        # Trim the audio to the correct length
        l_a = a.shape[0]
        ss = int(audio_clip_length * sr)
        if l_a == ss:
          synth_audio = a[0:ss]
        else:
          l_st = np.random.randint(0, l_a - ss)
          synth_audio = a[l_st:l_st + ss]
       
    elif len(transitions_list) > 0:
        synth_audio, synth_audio_seg_samples = create_template_audio_clip(audio_clip_length, samples_list, segments, sr)
        for i in range(len(transitions_list)):
            if transitions_list[i][0]['type'] == "normal":
                apply_normal_fade_out(synth_audio, transitions_list[i], synth_audio_seg_samples[i], sr)
                apply_normal_fade_in(synth_audio, transitions_list[i], synth_audio_seg_samples[i + 1], sr)
                
            elif transitions_list[i][0]['type'] == "cross-fade":
                apply_cross_fade_out(synth_audio, transitions_list[i], samples_list[i], segments[i], synth_audio_seg_samples[i], sr)
                apply_cross_fade_in(synth_audio, transitions_list[i], samples_list[i + 1], segments[i + 1], synth_audio_seg_samples[i + 1], sr)
        
        # Trim the audio to the correct length
        synth_audio = synth_audio[0:int(audio_clip_length * sr)]
    #print("synth_audio {}".format(synth_audio))
    #sf.write("synth_audio.wav", synth_audio, sr) 
    
    return synth_audio

def generate_multiclass_labels(audio_clip_length, transitions_list, class_list, sr = 22050.0, res = 220):
  """
  This function generates labels.
  `res` is in samples.
  """
  res_t = 220 / sr
  no_of_labels = int(np.ceil(audio_clip_length / res_t))

  #t = np.linspace(start=0.0, stop=audio_clip_length, num=no_of_labels, endpoint=False)
  #print(t)
  #print("t shape is {}".format(t.shape))



  class_list_01 = [0 if i == "speech" else 1 if i == "music" else 2 for i in class_list]
  class_list_01_opp = [1 if i == "speech" else 0 if i == "music" else 2 for i in class_list]
  #print("class_list_01 is {}".format(class_list_01))

  labels = np.zeros((no_of_labels, 2), dtype = np.int16)
  c = 0
  prev_point = 0

  for i in range(len(transitions_list)):
    #print(i)
    if class_list_01[c] != 2:
      labels[prev_point:int(transitions_list[i][1] / res_t), class_list_01[c]] = 1
    # Convert the fades into multiclass points.
    # The previous fade out should only be considered if it is not the first sample.
    if (True):
      if (transitions_list[i][0]['type'] == "cross-fade"):
        end_sample = int(transitions_list[i][1] / res_t) + int(transitions_list[i][0]['f_out_dur'] / res_t)
        start_sample = int(transitions_list[i][1] / res_t)
        #print("start_sample is {}".format(start_sample))
        #print("end_sample is {}".format(end_sample))
        if class_list_01[c] != 2:
          labels[start_sample:end_sample, class_list_01[c]] = 1

        start_sample = int(transitions_list[i][1] / res_t) - int(transitions_list[i][0]['f_in_dur'] / res_t)
        end_sample = int(transitions_list[i][1] / res_t)
        #print("start_sample is {}".format(start_sample))
        #print("end_sample is {}".format(end_sample))
        if class_list_01[c + 1] != 2:
          labels[start_sample:end_sample, class_list_01[c + 1]] = 1

      elif (transitions_list[i][0]['type'] == "normal"):
        start_sample = int(transitions_list[i][1] / res_t) - int(transitions_list[i][0]['time_gap'] / res_t)
        end_sample = int(transitions_list[i][1] / res_t)
        if class_list_01[c] != 2:
          labels[start_sample:end_sample, class_list_01[c]] = 0

      else:
        print("Hmmm... Something is wrong in the type of transitions...")


    # if (i != len(transitions_list) - 1):
    #   end_sample = int(transitions_list[i][1] / res_t) + int(transitions_list[i]['fade_in_dur'] / res_t)
    #   start_sample = int(transitions_list[i][1] / res_t)
    #   labels[start_sample:end_sample][class_list_01_opp[c]] = 1

    prev_point = int(transitions_list[i][1] / res_t)
    c += 1

  if class_list_01[c] != 2:
    labels[prev_point:no_of_labels,class_list_01[c]] = 1
  
  return labels


def generate_sed_eval_labels(audio_clip_length, transitions_list, class_list, sr = 22050.0, res = 220):
  prev_point = 0.0

  labels = []
  c = 0

  for i in range(len(transitions_list)):
    if (transitions_list[i][0]['type'] == "cross-fade"):
      end_point = transitions_list[i][1]
      end_point += transitions_list[i][0]['f_out_dur']
      labels.append((prev_point, end_point, class_list[c]))

    elif (transitions_list[i][0]['type'] == "normal"):
      end_point = transitions_list[i][1]
      end_point -= transitions_list[i][0]['time_gap']
      labels.append((prev_point, end_point, class_list[c]))

    else:
        print("Hmmm... Something is wrong in the type of transitions...")

    prev_point = transitions_list[i][1]
    c += 1

  labels.append((prev_point, audio_clip_length, class_list[c]))

  return labels

def synthesise_combined_audio_examples(no_of_examples, mel_dir, val_dir, test_dir, sr = 22050, audio_clip_length = 8.0, batch_size = 1, offset = 0):
    """
    This function synthesised audio examples and stores them in a directory.
    """           

    count_old = 0
    count_mixed = 0

    no_of_batches = int(np.floor(no_of_examples / batch_size))

    for b in range(no_of_batches):

      l = int(audio_clip_length * sr)

      res_t = 220 / sr
      no_of_labels = int(np.ceil(audio_clip_length / res_t))
      labels = np.zeros((no_of_labels, 2), dtype=np.int16)

      c = random.choice(["old", "mixed"])
      # c = "mixed"

      if c == "mixed":
        count_mixed += 1
        # print('c:', c)
        synth_audio, transition = create_mixed_audio_clip(audio_clip_length=audio_clip_length)
        synth_audio = librosa.util.normalize(synth_audio)
        
        labels[:, 0:2] = generate_mixed_multiclass_labels(transition, audio_clip_length=audio_clip_length)

      elif c == "old":
        count_old += 1
        # print('c:', c)
        p = create_random_transition_points(audio_clip_length, 1.0) # Create random transition points.
        
        transitions_list = create_transition_list(p) # Create a list of transitions.
        
        class_list = create_class_list(len(p) + 1) # Create a list of alternating classes for the segments.

                    
        samples_list = create_samples_list(class_list, music_sounds, speech_sounds, noise_sounds) # Create a list of randomly selected sounf files from the `music_sounds` and `speech_sounds` folder
        
        segment_lengths = get_segment_lengths(transitions_list, audio_clip_length)
        
        random_segments = get_random_segments(samples_list, segment_lengths)
        
        synth_audio = create_audio_clip(audio_clip_length, transitions_list, samples_list, random_segments, 22050)
        synth_audio = librosa.util.normalize(synth_audio)           
        
        labels[:, 0:2] = generate_multiclass_labels(audio_clip_length, transitions_list, class_list)

      else:
        print("\n\n\n Uncountered unexpected choice between old and mixed!!!! \n\n\n")


          #np.save(n_DbS_label, DbS_label)
      # multi_to_cat(labels)
      mel_id = 'mel-id-' + str(b + 1 + offset)
      
      choice_dir = random.choices(['train', 'val', 'test'], weights=[0.8, 0.2, 0.1], k=1)
      if choice_dir[0] == 'train':
        pp = mel_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'
      elif choice_dir[0] == 'val':
        pp = val_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'
      else:
        pp = test_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'

      if not os.path.isdir(pp):
        os.mkdir(pp)

      #synth_audio = synth_audio.astype(np.float32)
      # sf.write(pp + 'mel-id-' + str(b + 1 + offset) + '.wav', synth_audio , sr)
      M = get_log_melspectrogram(synth_audio)


      np.save(pp + 'mel-id-' + str(b + 1 + offset) + '.npy', M.T)
      np.save(pp + 'mel-id-label-' + str(b + 1 + offset) + '.npy', labels)

    return count_old, count_mixed

def synthesise_examples_OF(music_dir, speech_dir, mel_dir, sr = 22050, audio_clip_length = 8.0, batch_size = 1, offset = 0):
  """
  This function synthesised audio examples and stores them in a directory.
  """           
  example_len = int(sr * audio_clip_length)

  batch_size = 1
  no_of_labels = 802

  b = 0

  for s in speech_sounds:
    audio, _ = sf.read(s)
    if (audio.ndim > 1):
        in_signal_mono = librosa.to_mono(audio.T)
        audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(audio, orig_sr=_, target_sr=22050)
        audio = np.copy(in_signal_22k)

    audio_len = audio.shape[0]
    if audio_len < example_len:
      continue
    else:
      no_of_steps = int(audio_len // example_len)
      i = 0
      for i in range(no_of_steps):
        synth_audio = audio[example_len*i:example_len*(i + 1)]
        batch_audio = np.reshape(synth_audio, (1, example_len))
        synth_audio = librosa.util.normalize(synth_audio)
        labels = np.zeros((no_of_labels, 2), dtype = np.int16)
        labels[:, 0] = 1
          
        mel_id = 'mel-id-' + str(b + 1 + offset)
        M = get_log_melspectrogram(synth_audio)

        pp = mel_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'

        if not os.path.isdir(pp):
          os.mkdir(pp)

        np.save(pp + 'mel-id-' + str(b + 1 + offset) + '.npy', M.T)
        np.save(pp + 'mel-id-label-' + str(b + 1 + offset) + '.npy', labels)
        
        b += 1

      if audio_len >= no_of_steps * example_len:
        synth_audio = audio[audio_len - example_len:audio_len]
        synth_audio = librosa.util.normalize(synth_audio)

        labels = np.zeros((no_of_labels, 2), dtype = np.int16)
        labels[:, 0] = 1

        mel_id = 'mel-id-' + str(b + 1 + offset)
        M = get_log_melspectrogram(synth_audio)

        pp = mel_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'

        if not os.path.isdir(pp):
          os.mkdir(pp)

        np.save(pp + 'mel-id-' + str(b + 1 + offset) + '.npy', M.T)
        np.save(pp + 'mel-id-label-' + str(b + 1 + offset) + '.npy', labels)
        
        b += 1

  for s in music_sounds:
    audio, _ = sf.read(s)
    if (audio.ndim > 1):
        in_signal_mono = librosa.to_mono(audio.T)
        audio = np.copy(in_signal_mono)
    if _ != 22050:
        in_signal_22k = librosa.resample(audio, orig_sr=_, target_sr=22050)
        audio = np.copy(in_signal_22k)

    audio_len = audio.shape[0]
    if audio_len < example_len:
      continue
    else:
      no_of_steps = int(audio_len // example_len)
      i = 0
      for i in range(no_of_steps):
        synth_audio = audio[example_len * i:example_len * (i + 1)]
        synth_audio = librosa.util.normalize(synth_audio)

        labels = np.zeros((no_of_labels, 2))
        labels[:, 1] = 1

          
        mel_id = 'mel-id-' + str(b + 1 + offset)
        M = get_log_melspectrogram(synth_audio)

        pp = mel_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'

        if not os.path.isdir(pp):
          os.mkdir(pp)

        np.save(pp + 'mel-id-' + str(b + 1 + offset) + '.npy', M.T)
        np.save(pp + 'mel-id-label-' + str(b + 1 + offset) + '.npy', labels)
        
        b += 1

      if audio_len >= no_of_steps * example_len:
        synth_audio = audio[audio_len - example_len:audio_len]
        synth_audio = librosa.util.normalize(synth_audio)
        labels = np.zeros((no_of_labels, 2), dtype = np.int16)
        labels[:, 1] = 1

          
        mel_id = 'mel-id-' + str(b + 1 + offset)
        labels = get_log_melspectrogram(synth_audio)

        pp = mel_dir + '/block-id-' + str(get_block_id(mel_id)) + '/'

        if not os.path.isdir(pp):
          os.mkdir(pp)

        np.save(pp + 'mel-id-' + str(b + 1 + offset) + '.npy', M.T)
        np.save(pp + 'mel-id-label-' + str(b + 1 + offset) + '.npy', labels)
        
        b += 1


speech_sounds = speech_list
music_sounds = music_list
noise_sounds = noise_list
print(len(speech_sounds))
print(len(music_sounds))
print(len(noise_sounds))

# Location of the directory to store the audio examples.

"""
Split the data synthesis into parts of 5120 examples. 
In the below code, i ranges from 0 to 8, and thus creating 5120 * 8 = 40960 examples.
If you are synthesising the val set, a range from 0 to 1 might be appropriate.
"""
mel_dir = "./musan/Mel_Files_0425_num_5120_music_0.4_speech_0.4_noise_0.2"
val_dir = "./musan/val_0425_num_5120_music_0.4_speech_0.4_noise_0.2"
test_dir = "./musan/test_0425_num_5120_music_0.4_speech_0.4_noise_0.2"
num_samples = 5120

try:
  os.makedirs(mel_dir, exist_ok = True)
  os.makedirs(val_dir, exist_ok = True)
  os.makedirs(test_dir, exist_ok = True)
  print("train/val/test Directory created successfully" )
except OSError as error:
    print("Directory can not be created")

for i in range(0, 8):
  count_old, count_mixed = synthesise_combined_audio_examples(num_samples, mel_dir, val_dir, test_dir, offset = num_samples * i, audio_clip_length = 8.0)
print('train / val / test data propare success.')

