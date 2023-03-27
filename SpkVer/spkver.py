# Flask app to run speaker verification
from flask import Flask, jsonify, request
import logging
import subprocess
import datetime
import os

from speechbrain_spkver import SpeakerVerification
spk_ver = SpeakerVerification()

FILE_UPLOAD_DIR="/home/smtlab/VaulterVoice/SpkVer/uploads"
ENABLE_TIMESTAMP=True
SAMPLING_RATE=16000

logger = logging.getLogger("Speaker Verification")
logger.setLevel(logging.INFO)

app = Flask(__name__)

def run_spk_ver(input_file, request_data):
    '''
    run spk verification
    '''
    logger.info(f"Running speaker verification function")
    print(f"Running speaker verification function")
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
        print(f"File saved at {file_save_path}")
        spk_id = request_data["spk_id"] if "spk_id" in request_data else "Tmp"
        logger.info(f"spk_id: {spk_id}")
        print(f"spk_id: {spk_id}")
        if spk_id == "Tmp":
            return jsonify(status="failure", reason='spk_id not found')
        try:
            ver_status = spk_ver.verify_spk_speechbrain(spk_id, file_save_path, test_type="audio")
            logger.info(f"ver_status: {ver_status}")
            print(f"ver_status: {ver_status}")
            return jsonify(status="success", verification_status=ver_status), (tmp_dir, file_save_path)
        except Exception as e:
            return jsonify(status="failure", reason='speaker verification failed')

def run_spk_add(input_file, request_data):
    '''
    run spk add
    '''
    logger.info(f"Running speaker add function")
    print(f"Running speaker add function")
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
        print(f"File saved at {file_save_path}")
        spk_id = request_data["spk_id"] if "spk_id" in request_data else "Tmp"
        logger.info(f"spk_id: {spk_id}")
        print(f"spk_id: {spk_id}")
        if spk_id == "Tmp":
            return jsonify(status="failure", reason='spk_id not found')
        try:
            spk_ver.add_spk(spk_id, file_save_path)
            return jsonify(status="success"), (tmp_dir, file_save_path)
        except Exception as e:
            return jsonify(status="failure", reason='audio adding failed')

@app.route('/verify', methods=['POST'], strict_slashes=False)
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

@app.route('/add', methods=['POST'], strict_slashes=False)
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