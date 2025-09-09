python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
