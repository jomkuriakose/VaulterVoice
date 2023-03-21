from flask import Flask, jsonify, request, render_template
import logging
import traceback
import subprocess

from spkVerifi import SpeakerVerification
spk_ver = SpeakerVerification()

logger = logging.getLogger("Speaker Verification")
logger.setLevel(logging.INFO)

app = Flask(__name__)

def run_spk_ver():
    '''
    '''
    

@app.route('/', methods=['GET', 'POST'], strict_slashes=False)
def spkver():
    try:
        json_data = request.get_json()
        logger.info(f"request received")
        out, tmp_dir = run_spk_ver(json_data)
        return out
    except Exception as err:
        logger.error(traceback.format_exc())
        return jsonify(status="failure", reason=str(err))
    finally:
        # release resources and remove temp files
        # clear temp files in a background process
        if tmp_dir is not None:
            subprocess.Popen(["rm","-rf",tmp_dir])

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.    
    app.run(host='0.0.0.0', port=6789)