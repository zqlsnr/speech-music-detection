"""
Remove all silences and resample the sound files.
"""
import os
from shutil import copyfile
import soundfile as sf
import glob
music_list = glob.glob('./musan/music/**/*.wav', recursive = True)
speech_list = glob.glob('./musan/speech/**/*.wav', recursive = True)
noise_list = glob.glob('./musan/noise/**/*.wav', recursive = True)


for sound in music_list:
  temp_file = sound.replace('.wav', '-t.wav')
  os.system("sox {} {} rate 22050 silence -l 1 0.1 1% -1 0.1 1%".format(sound, temp_file))
  copyfile(temp_file, sound)
  os.remove(temp_file)

for sound in speech_list:
  temp_file = sound.replace('.wav', '-t.wav')
  os.system("sox {} {} rate 22050 silence -l 1 0.1 1% -1 0.1 1%".format(sound,temp_file))
  copyfile(temp_file, sound)
  os.remove(temp_file)

for sound in noise_list:
  temp_file = sound.replace('.wav', '-t.wav')
  os.system("sox {} {} rate 22050 silence -l 1 0.1 1% -1 0.1 1%".format(sound,temp_file))
  copyfile(temp_file, sound)
  os.remove(temp_file)
# If there are files shorter than 9.1 s, loop them 4 times to increase their lengths.

for sound in noise_list:
  d, sr = sf.read(sound)
  t = float(d.shape[0]) / sr
  if t < 9.1:
    temp_file = sound.replace('.wav', '-t.wav')
    os.system("sox {} {} repeat 4".format(sound,temp_file))
    copyfile(temp_file, sound)
    os.remove(temp_file)

print(len(music_list))
print(len(speech_list))
print(len(noise_list))

print('all success.')