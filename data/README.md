Install bokel prerequisites for data generation

```
git clone https://github.com/Maluuba/bokeh.git

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
source ~/.bashrc
nvm list-remote
nvm install v11.15.0

npm install -g phantomjs-prebuilt

cd bokeh
git checkout 437886fe72d62f43d72249d92e4f8e9d5a66fd11
cd bokehjs
npm install

cd ../
pip install setuptools==65.0.0
python setup.py install --build-js
cd ../
```

Install figureQA
```
git clone https://github.com/Maluuba/FigureQA.git
cd FigureQA
pip install -r requirements.txt

pip install jinja2==3.0.1
```

Running generation:

```
export OPENSSL_CONF=/etc/ssl/

python figureqa/generation/generate_dataset.py ./config/custom_config.yaml 
```