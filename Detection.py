# -*- coding: utf-8 -*-

import soundfile as sf
import math
import numpy as np
import librosa
import os
import time
from model_new import CRNN_Model
from utils import smooth_output, preds_to_se, get_log_melspectrogram, convert_seconds_to_hms

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



"""
Make predictions for full audio.
"""


def mk_preds_fa(audio_path, hop_size=6.0, discard=1.0, win_length=8.0, sampling_rate=22050):
    in_signal, in_sr = sf.read(audio_path, dtype='float32')

    if in_signal.ndim > 1:
        in_signal_mono = librosa.to_mono(in_signal.T)
        in_signal = np.copy(in_signal_mono)
    # Resample the audio file.
    in_signal_22k = librosa.resample(in_signal, orig_sr=in_sr, target_sr=sampling_rate)
    in_signal = np.copy(in_signal_22k)

    # Pad the input signal if it is shorter than 8 s.

    if in_signal.shape[0] < int(8.0 * sampling_rate):
        pad_signal = np.zeros((int(8.0 * sampling_rate)))
        pad_signal[:in_signal.shape[0]] = in_signal
        in_signal = np.copy(pad_signal)

    audio_clip_length_samples = in_signal.shape[0]
    print('audio_clip_length_samples is {}'.format(audio_clip_length_samples))

    hop_size_samples = 220 * 602 - 1

    win_length_samples = 220 * 802 - 1

    n_preds = int(math.ceil((audio_clip_length_samples - win_length_samples) / hop_size_samples)) + 1

    in_signal_pad = np.zeros((n_preds * hop_size_samples + 200 * 220))

    in_signal_pad[0:audio_clip_length_samples] = in_signal

    preds = np.zeros((n_preds, 802, 2))

    # Split the predictions into batches of size batch_size.

    batch_size = 128

    n_batch = n_preds // batch_size

    for i in range(n_batch):
        # mss_batch = np.zeros((batch_size, 802, 80))
        mss_batch = np.zeros((batch_size, 802, 100))
        for j in range(batch_size):
            seg = in_signal_pad[(i * batch_size + j) * hop_size_samples:((
                                                                                 i * batch_size + j) * hop_size_samples) + win_length_samples]
            seg = librosa.util.normalize(seg)
            mss = get_log_melspectrogram(seg)
            M = mss.T
            mss_batch[j, :, :] = M

        preds[i * batch_size:(i + 1) * batch_size, :, :] = (model.predict(mss_batch) >= (0.5, 0.5)).astype(float)

    if n_batch * batch_size < n_preds:
        i = n_batch
        # mss_batch = np.zeros((n_preds - n_batch * batch_size, 802, 80))
        mss_batch = np.zeros((n_preds - n_batch * batch_size, 802, 100))
        for j in range(n_preds - n_batch * batch_size):
            seg = in_signal_pad[(i * batch_size + j) * hop_size_samples:((
                                                                                 i * batch_size + j) * hop_size_samples) + win_length_samples]
            seg = librosa.util.normalize(seg)
            mss = get_log_melspectrogram(seg)
            M = mss.T
            mss_batch[j, :, :] = M

        preds[i * batch_size:n_preds, :, :] = (model.predict(mss_batch) >= (0.5, 0.5)).astype(float)

    preds_mid = np.copy(preds[1:-1, 100:702, :])

    preds_mid_2 = preds_mid.reshape(-1, 2)

    if preds.shape[0] > 1:
        oa_preds = preds[0, 0:702, :]  # oa stands for overall predictions

    else:
        oa_preds = preds[0, 0:802, :]  # oa stands for overall predictions

    oa_preds = np.concatenate((oa_preds, preds_mid_2), axis=0)

    if preds.shape[0] > 1:
        oa_preds = np.concatenate((oa_preds, preds[-1, 100:, :]), axis=0)

    return oa_preds


def delete_file(file_path):
    if os.path.exists(file_path):
        # 删除文件  
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除")
    else:
        print(f"文件 {file_path} 不存在")


if __name__ == '__main__':
    wav_name = ''
    start_time = time.time()
    test_audio = wav_name
    output_path = os.path.basename(wav_name).split(".")[0] + '.txt'

    # load model
    m = './model_result/best2-epoch-30-new-model-GRU-40-0419-num_5120_music_0.4_speech_0.4_noise_0.2.h5'

    model = CRNN_Model()

    model.load_weights(m)

    ss, _ = sf.read(test_audio)
    oop = mk_preds_fa(test_audio)

    p_smooth = smooth_output(oop.T, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6)
    p_smooth = p_smooth.T
    see = preds_to_se(p_smooth, audio_clip_length=ss.shape[0] / 22050.0)

    n_label = output_path
    end_time = time.time()
    print('cost time:', end_time - start_time, "秒")

    with open(n_label, 'w') as fp:
        fp.write('\n'.join('{}\t{}\t{}'.format(convert_seconds_to_hms(round(x[0], 5)),
                                               convert_seconds_to_hms(round(x[1], 5)), x[2]) for x in see))
