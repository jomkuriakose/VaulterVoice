# Speaker verfication using Microsoft WavLM + X-Vectors
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import librosa
import numpy as np
import os
import shutil
import traceback

threshold = 0.80 #0.86 suggested in huggingface
sampling_rate = 16000
# parameters for sil trimming -- Energy based
sil_threshold_energy=0.5
min_silence_duration=0.5
# parameters for sil trimming -- Decibel based
sil_threshold_decibel=20
# location of speaker files
spk_ref_files_folder = './spk_audio/'

class SpeakerVerification:

    def __init__(self, model=None):
        # Loading the models
        try:
            if model is None or model == 'base-plus':
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
                self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
            elif model == 'large':
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
                self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-large')
            elif model == 'base':
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
                self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
            else:
                print(f"Error:: model name not found: {model}")
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")
        # Defining the speaker ref files dictionary
        self.spk_ref = {}
        self.__load_spk_ref()
        # print(self.spk_ref)
    
    def __load_audio_np(self, filename):
        try:
            # Load audio file
            audio, sr = librosa.load(filename, sr=sampling_rate, mono=True)
            # Convert audio to numpy array
            audio_np = np.array(audio)
            # Trim start and end silences
            # audio_np = self.__trim_sil_decibel(audio_np)
            audio_np = self.__remove_silence(self.__normalize_audio(audio_np))
            return audio_np
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")
    
    # Function to normalize an audio signal to a specified maximum amplitude
    def __normalize_audio(self, audio_signal, max_amplitude=0.5):
        max_value = np.max(np.abs(audio_signal))
        normalized_audio = audio_signal * (max_amplitude / max_value)
        return normalized_audio

    # Function to remove beginning and end silence regions from an audio signal
    def __remove_silence(self, audio_signal, threshold=0.01, pad_duration=0.5):
        energy = librosa.feature.rms(y=audio_signal)
        frames = np.nonzero(energy > threshold)
        indices = librosa.frames_to_samples(frames)[1]
        trimmed_audio = audio_signal[indices[0]:indices[-1]]
        trimmed_audio = np.pad(trimmed_audio, int(pad_duration * audio_signal.shape[0]), mode='reflect')
        return trimmed_audio

    # Not working some error - has to fix it
    def __trim_sil_energy(self, audio_data):
        energy = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)
        # Find non-silent start and end points
        start_idx = 0
        while energy[0][start_idx] < sil_threshold_energy:
            start_idx += 1
        end_idx = len(energy[0]) - 1
        while energy[0][end_idx] < sil_threshold_energy:
            end_idx -= 1
        # Check if the duration of the trimmed audio is greater than the minimum silence duration
        start_time = librosa.samples_to_time(start_idx * 512, sr=sampling_rate)
        end_time = librosa.samples_to_time((end_idx + 1) * 512, sr=sampling_rate)
        trimmed_duration = end_time - start_time
        if trimmed_duration < min_silence_duration:
            return audio_data
        # Trim the audio data
        trimmed_audio_data = audio_data[start_idx * 512 : (end_idx + 1) * 512]
        return trimmed_audio_data

    def __trim_sil_decibel(self, audio_data):
        # Trim the silences from the beginning and end of the audio signal
        trimmed_audio_data, _ = librosa.effects.trim(y=audio_data, top_db=sil_threshold_decibel, frame_length=2048, hop_length=512)
        return trimmed_audio_data
    
    def __load_spk_ref(self, spk_fold=None):
        if spk_fold is None:
            self.spk_files_fold = spk_ref_files_folder
        else:
            self.spk_files_fold = spk_fold
        if not os.path.exists(self.spk_files_fold):
            os.makedirs(self.spk_files_fold)
        else:
            try:
                for spker in os.listdir(self.spk_files_fold):
                    if spker.startswith('.'):
                        continue
                    self.spk_ref[spker] = []
                    for audio in os.listdir(f"{self.spk_files_fold}/{spker}"):
                        if audio.startswith('.'):
                            continue
                        self.spk_ref[spker].append(self.__load_audio_np(f"{self.spk_files_fold}/{spker}/{audio}"))
            except Exception as e:
                print(traceback.format_exc())
                print(f"Error:: speaker ref loading failed")

    def __spk_sim(self, ref_audio, test_audio, input_type=None):
        # print(input_type)
        if input_type is None or input_type == 'file':
            audio = [self.__load_audio_np(ref_audio), self.__load_audio_np(test_audio)]
        elif input_type == 'array':
            if type(ref_audio) != np.ndarray:
                try:
                    ref_audio = np.array(ref_audio)
                except Exception as e:
                    print(traceback.format_exc())
                    print(f"Error:: {e}")
            if type(test_audio) != np.ndarray:
                try:
                    test_audio = np.array(test_audio)
                except Exception as e:
                    print(traceback.format_exc())
                    print(f"Error:: {e}")
            audio = [ref_audio, test_audio]
        else:
            print('Error::  input type not supported')
        # print('Begin')
        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, padding=True, return_tensors="pt")
        # print('End')
        embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        # the resulting embeddings can be used for cosine similarity-based retrieval
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings[0], embeddings[1])
        print(f"similarity: {similarity}")
        return similarity

    def add_spk(self, spk_id, audio_file):
        if spk_id in self.spk_ref.keys():
            self.spk_ref[spk_id].append(self.__load_audio_np(audio_file))
            shutil.copyfile(audio_file, f"{self.spk_files_fold}/{spk_id}")
        else:
            self.spk_ref[spk_id] = [self.__load_audio_np(audio_file)]
            os.makedirs(f"{self.spk_files_fold}/{spk_id}", exist_ok = True)
            shutil.copyfile(audio_file, f"{self.spk_files_fold}/{spk_id}")
    
    def verify_spk(self, spk_id, test_audio):
        if spk_id not in self.spk_ref.keys():
            print(f"Error:: audio for {spk_id} not found in DB")
        else:
            for i in range(0, len(self.spk_ref[spk_id])):
                if self.__spk_sim(self.spk_ref[spk_id][i], self.__load_audio_np(test_audio), input_type='array') > threshold:
                    continue
                else:
                    return False
            return True
