from flask import Flask, jsonify, request, render_template
import logging
import traceback
import subprocess
import datetime
import time
import librosa
import os

from spkVerifi import SpeakerVerification
spk_ver = SpeakerVerification()

FILE_UPLOAD_DIR="/music/jom/Hackathon/VaulterVoice/SpkVer/uploads"
ENABLE_TIMESTAMP=True
SAMPLING_RATE=16000

logger = logging.getLogger("Speaker Verification")
logger.setLevel(logging.INFO)

app = Flask(__name__)

def run_spk_ver_arr(request_data):
    st = time.time()
    tmp_dir = None
    with app.app_context():
        spkid = json_data["spk_id"]
        audio_arr = json_data["audio"]

def run_spk_ver(input_file, request_data):
    tmp_dir = None
    with app.app_context():
        timestamp=""
        if ENABLE_TIMESTAMP:
            timestamp=str(datetime.datetime.now())
            for spl_char in ['-', ' ', ':', '.']:
                timestamp = timestamp.replace(spl_char, '_')
        filename, ext = None, None
        if "." in input_file.filename:
            filename, ext = input_file.filename.split(".")
        else:
            filename = input_file.filename
        tmp_dir=f"tmp/data/data_{timestamp}"
        file_save_path = os.path.join(FILE_UPLOAD_DIR, filename + "_" + timestamp + "." + ext)
        input_file.save(file_save_path)
        logger.info(f"File saved at {file_save_path}")
        spk_id = request_data["spk_id"] if "spk_id" in request_data else None
        if spk_id is None:
            return jsonify(status="failure", reason='spk_id not found')
        model = request_data["model"] if "model" in request_data else "speechbrain"
        try:
            if model == 'speechbrain':
                ver_status = spk_ver.verify_spk_speechbrain(spk_id, file_save_path)
                return jsonify(status="success", verification_status=ver_status), (tmp_dir, file_save_path)
            elif model == 'wavlm':
                try:
                    data, samplerate = librosa.load(file_save_path, sr=SAMPLING_RATE)
                except Exception as e:
                    return jsonify(status="failure", reason='file not found')
                ver_status = spk_ver.verify_spk_wavlm(spk_id, data, test_type == 'array')
                return jsonify(status="success", verification_status=ver_status), (tmp_dir, file_save_path)
            else:
                return jsonify(status="failure", reason='model not found')
        except Exception as e:
            return jsonify(status="failure", reason='speaker verification failed')

def run_spk_add(input_file, request_data):
    tmp_dir = None
    with app.app_context():
        timestamp=""
        if ENABLE_TIMESTAMP:
            timestamp=str(datetime.datetime.now())
            for spl_char in ['-', ' ', ':', '.']:
                timestamp = timestamp.replace(spl_char, '_')
        filename, ext = None, None
        if "." in input_file.filename:
            filename, ext = input_file.filename.split(".")
        else:
            filename = input_file.filename
        tmp_dir=f"tmp/data/data_{timestamp}"
        file_save_path = os.path.join(FILE_UPLOAD_DIR, filename + "_" + timestamp + "." + ext)
        input_file.save(file_save_path)
        logger.info(f"File saved at {file_save_path}")
        spk_id = request_data["spk_id"] if "spk_id" in request_data else None
        if spk_id is None:
            return jsonify(status="failure", reason='spk_id not found')
        try:
            spk_ver.add_spk(spk_id, file_save_path)
            return jsonify(status="success"), (tmp_dir, file_save_path)
        except Exception as e:
            return jsonify(status="failure", reason='audio adding failed')

@app.route('/verify', methods=['GET', 'POST'], strict_slashes=False)
def spkver():
    tmp_files = []
    request_data = request.get_json(force=True, silent=True)
    if request_data is None:
        request_data = request.values
    logger.debug("Request received")
    files = request.files
    input_file = None
    try:
        input_file = files.get('file')
    except Exception as err:
        logger.error(err)
        return jsonify(status='failure', reason=f"Unsupported input. {err}")
    try:
        out, tmp_files = run_spk_ver(input_file, request_data)
        return out
    except Exception as err:
        logger.error(err)
        return jsonify(status="failure", reason=str(err))
    finally:
        # clear temp files in a background process
        for tmp_file in tmp_files:
            if tmp_file is not None:
                subprocess.Popen(["rm","-rf",tmp_file])

@app.route('/add', methods=['GET', 'POST'], strict_slashes=False)
def spkadd():
    tmp_files = []
    request_data = request.get_json(force=True, silent=True)
    if request_data is None:
        request_data = request.values
    logger.debug("Request received")
    files = request.files
    input_file = None
    try:
        input_file = files.get('file')
    except Exception as err:
        logger.error(err)
        return jsonify(status='failure', reason=f"Unsupported input. {err}")
    try:
        out, tmp_files = run_spk_add(input_file, request_data)
        return out
    except Exception as err:
        logger.error(err)
        return jsonify(status="failure", reason=str(err))
    finally:
        # clear temp files in a background process
        for tmp_file in tmp_files:
            if tmp_file is not None:
                subprocess.Popen(["rm","-rf",tmp_file])

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.    
    app.run(host='0.0.0.0', port=6789)