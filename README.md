# xltts-server

This code is a code copy uploaded separately for items that have modified the source code in [FastSpeech2 implementation](https://github.com/ming024/FastSpeech2).

It consists of FastSpeech2 + FastAPI + KoG2P.
Therefore, each setting is required, but KoG2P is built-in and does not require installation.

The zipped full code is [here](https://drive.google.com/file/d/1o-daPiF-pzW0uNXy3p63zW1HDOHDx3n2/view?usp=sharing)

# How to install
This project assumes that the conda environment is prepared.

The Python version used is python 3.9 and the CUDA version is 12.0, but 11.6 or 11.7 is recommended.

## Caution
- Be sure to prepare as it will not run without the CUDA environment being built.
- If there are some problems installing dependencies, try to install
 the dependencies in requirements.txt manually.


## 1. PyTorch Installation
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 2. FastAPI Installation
```
pip install "fastapi[all]"
pip install fastapi-utils
```

## 3. scikit-learn Installation
```
pip install -U scikit-learn
```

## 4. openai Installation
```
pip install openai
```

## 5. Other dependencies Installation
```
pip install -r requirements.txt

```

## 6. Run
```
uvicorn main:app --reload
```

# Test
You can simply check using curl.

```
curl --location 'http://127.0.0.1:8000/tts/?text=hello&is_kor=0'
```

Or you can use your internet browser.
```
http://127.0.0.1:8000/tts/?text=hello&is_kor=0
```

Success is achieved if the output is similar to the following:
```
{"key":"oMgpOriu5WM6XTF8"}
```

## Result
**교수님도 사람이야**

![image](https://github.com/noelvalent/xltts-server/blob/main/output/result/LJSpeech/%EA%B5%90%EC%88%98%EB%8B%98%EB%8F%84%20%EC%82%AC%EB%9E%8C%EC%9D%B4%EC%95%BC.png?raw=true)

Audio file: [link](https://github.com/noelvalent/xltts-server/blob/main/output/result/LJSpeech/Y5SiU91h0rjF2r7k.wav?raw=true)

