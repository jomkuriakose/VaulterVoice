# Flask app to run speaker verification
from flask import Flask, jsonify, request
import logging

from textProc import TextNormalizer
text_norm = TextNormalizer()

logger = logging.getLogger("Text Norm")
logger.setLevel(logging.INFO)

app = Flask(__name__)

def text_norm_func(request_data):
    '''
    Text Norm
    '''
    logger.info(f"Running Text Norm function")
    print(f"Running Text Norm function")
    with app.app_context():
        text = request_data["text"] if "text" in request_data else "Tmp"
        lang = request_data["lang"] if "lang" in request_data else "Tmp"
        logger.info(f"spk_id: {text}, lang: {lang}")
        print(f"spk_id: {text}, lang: {lang}")
        if text == "Tmp" or lang == "Tmp":
            return jsonify(status="failure", reason='input not complete. text and lang expected')
        try:
            ver_status, output = text_proc.number_to_words(text, lang)
            logger.info(f"ver_status: {ver_status}, output: {output}")
            print(f"ver_status: {ver_status}, output: {output}")
            if ver_status:
                return jsonify(status="success", output=output)
            else:
                return jsonify(status="failure", reason=output)
        except Exception as e:
            return jsonify(status="failure", reason='text norm failed')

@app.route('/', methods=['GET, POST'], strict_slashes=False)
def textnorm():
    request_data = request.get_json(force=True, silent=True)
    if request_data is None:
        request_data = request.values
    logger.debug("Request received")
    try:
        out = text_norm_func(request_data)
        return out
    except Exception as err:
        logger.error(err)
        return jsonify(status="failure", reason=str(err))

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.    
    app.run(host='0.0.0.0', port=6790)