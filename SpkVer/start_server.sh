source /music/jom/Hackathon/VaulterVoice/venv/bin/activate
USE_NNPACK=0 gunicorn -w 1 -b 127.0.0.1:6789 --access-logfile=- --error-logfile=- --timeout 600 'spkver:app'