# Speaker verification using Microsoft WavLM + X-Vectors
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import librosa
import numpy as np
import os
import shutil
import traceback
from speechbrain.pretrained import SpeakerRecognition

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
        self.model_class = None
        try:
            if model is None or model == 'speechbrain':
                self.speechbrain_verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
                self.model_class = 'speechbrain'
            elif model == 'wavlm-base-plus':
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
                self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
                self.model_class = 'wavlm'
            elif model == 'wavlm-large':
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
                self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-large')
                self.model_class = 'wavlm'
            elif model == 'wavlm-base':
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
                self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
                self.model_class = 'wavlm'
            else:
                print(f"Error:: model name not found: {model}")
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")
        # Defining the speaker ref files dictionary
        self.spk_ref = {}
        self.spk_ref_files = {}
        self.__load_spk_ref()
        # print(self.spk_ref)
        # print(self.spk_ref_files)
    
    def __load_audio_np(self, filename):
        try:
            # Load audio file
            audio, sr = librosa.load(filename, sr=sampling_rate, mono=True)
            # Convert audio to numpy array
            audio_np = np.array(audio)
            # Trim start and end silences
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
                    self.spk_ref_files[spker] = []
                    for audio in os.listdir(f"{self.spk_files_fold}/{spker}"):
                        if audio.startswith('.'):
                            continue
                        self.spk_ref[spker].append(self.__load_audio_np(f"{self.spk_files_fold}/{spker}/{audio}"))
                        self.spk_ref_files[spker].append(f"{self.spk_files_fold}/{spker}/{audio}")
            except Exception as e:
                print(traceback.format_exc())
                print(f"Error:: speaker ref loading failed")

    def __spk_sim_wavlm(self, ref_audio, test_audio, input_type=None):
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
    
    def __spk_sim_speechbrain(self, ref_audio, test_audio):
        try:
            score, prediction = self.speechbrain_verification.verify_files(ref_audio, test_audio)
            print(f"similarity: {score}")
            return score
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")

    def add_spk(self, spk_id, audio_file):
        if spk_id in self.spk_ref.keys():
            self.spk_ref[spk_id].append(self.__load_audio_np(audio_file))
            shutil.copyfile(audio_file, f"{self.spk_files_fold}/{spk_id}")
        else:
            self.spk_ref[spk_id] = [self.__load_audio_np(audio_file)]
            os.makedirs(f"{self.spk_files_fold}/{spk_id}", exist_ok = True)
            shutil.copyfile(audio_file, f"{self.spk_files_fold}/{spk_id}")
    
    def verify_spk_wavlm(self, spk_id, test_audio, test_type=None):
        if test_type is None or test_type == 'array':
            if spk_id not in self.spk_ref.keys():
                print(f"Error:: audio for {spk_id} not found in DB")
            else:
                for i in range(0, len(self.spk_ref[spk_id])):
                    if self.__spk_sim_wavlm(self.spk_ref[spk_id][i], test_audio, input_type='array') > threshold:
                        continue
                    else:
                        return False
                return True
        elif test_type == 'audio':
            if spk_id not in self.spk_ref.keys():
                print(f"Error:: audio for {spk_id} not found in DB")
            else:
                for i in range(0, len(self.spk_ref[spk_id])):
                    if self.__spk_sim_wavlm(self.spk_ref[spk_id][i], self.__load_audio_np(test_audio), input_type='array') > threshold:
                        continue
                    else:
                        return False
                return True
        else:
            print(f"Error:: invalid test type: {test_type}")

    def verify_spk_speechbrain(self, spk_id, test_audio):
        if spk_id not in self.spk_ref_files.keys():
            print(f"Error:: audio for {spk_id} not found in DB")
        else:
            for i in range(0, len(self.spk_ref[spk_id])):
                if self.__spk_sim_speechbrain(self.spk_ref[spk_id][i], test_audio) > threshold:
                    continue
                else:
                    return False
            return True
