# Speaker verification using Speech Brain with ECAPA-TDNN embeddings
# https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
import os
import shutil
import traceback
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import Pretrained

# location of speaker files
spk_ref_files_folder = './spk_audio/'

class SpeakerVerification:
    '''
    Class for SpeakerVerification
    '''
    def __init__(self):
        '''
        Initialization function to load the speechbrain pre-trained speaker verification model and speechbrain audio loader function
        '''
        try:
            # Speaker verification function
            self.speechbrain_verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
            # Audio file loader function
            self.speechbrain_data_loader = Pretrained.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")
        
        # Defining the speaker ref files dictionary
        self.spk_ref = {}
        self.spk_ref_files = {}
        self.__load_spk_ref()
    
    def __load_audio(self, filename):
        '''
        Load the audio to tensor using speechbrain load_audio function
        audio is normalized to 16k sampling rate
        '''
        try:
            # Load audio file
            audio = self.speechbrain_data_loader.load_audio(filename)
            return audio
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")
    
    def __load_spk_ref(self, spk_fold=None):
        '''
        Load the speaker files from each folder in the speaker files folder
        '''
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
                        self.spk_ref[spker].append(self.__load_audio(f"{self.spk_files_fold}/{spker}/{audio}"))
                        self.spk_ref_files[spker].append(f"{self.spk_files_fold}/{spker}/{audio}")
            except Exception as e:
                print(traceback.format_exc())
                print(f"Error:: speaker ref loading failed")

    def __spk_veri_speechbrain(self, ref_audio, test_audio, test_type=None):
        '''
        Function to verify if two audio files belong to the same speaker. Return True or False
        '''
        try:
            if test_type is None or test_type == 'tensor':
                score, prediction = self.speechbrain_verification.verify_batch(ref_audio, test_audio)
                return bool(prediction.tolist()[0][0])
            elif test_type == 'audio':
                score, prediction = self.speechbrain_verification.verify_files(ref_audio, test_audio)
                return bool(prediction.tolist()[0])
            else:
                print('Error::  input type not supported')
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error:: {e}")

    def add_spk(self, spk_id, audio_file):
        '''
        Function to add audio files as reference files for a speaker
        '''
        if spk_id in self.spk_ref.keys():
            self.spk_ref[spk_id].append(self.__load_audio(audio_file))
            shutil.copyfile(audio_file, f"{self.spk_files_fold}/{spk_id}")
        else:
            self.spk_ref[spk_id] = [self.__load_audio(audio_file)]
            os.makedirs(f"{self.spk_files_fold}/{spk_id}", exist_ok = True)
            shutil.copyfile(audio_file, f"{self.spk_files_fold}/{spk_id}")
    
    def verify_spk_speechbrain(self, spk_id, test_audio, test_type=None):
        '''
        Verify a test audio given speaker identity against which the audio has to be verified
        '''
        if test_type is None or test_type == 'tensor':
            if spk_id not in self.spk_ref.keys():
                print(f"Error:: audio for {spk_id} not found in DB")
            else:
                for i in range(0, len(self.spk_ref[spk_id])):
                    if self.__spk_veri_speechbrain(self.spk_ref[spk_id][i], test_audio) == True:
                        continue
                    else:
                        return False
                return True
        elif test_type == 'audio':
            if spk_id not in self.spk_ref.keys():
                print(f"Error:: audio for {spk_id} not found in DB")
            else:
                for i in range(0, len(self.spk_ref[spk_id])):
                    if self.__spk_veri_speechbrain(self.spk_ref[spk_id][i], self.__load_audio(test_audio)) == True:
                        continue
                    else:
                        return False
                return True
        else:
            print(f"Error:: invalid test type: {test_type}")
