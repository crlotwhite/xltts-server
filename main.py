from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi_utils.tasks import repeat_every

from server.cache import Cache
from synthesize import run_synthesizer
from server.gpt3 import chat


app = FastAPI()
cache = Cache()

app.mount("/output", StaticFiles(directory="output"), name="output")

def clean_directory():
    pass
    # import os
    #
    # for file in os.listdir('audios'):
    #     os.remove(f'audios/{file}')
    #     print(f'remove {file}')


@app.on_event('startup')
def start_up():
    clean_directory()
    print('Server Start!')


@app.on_event('shutdown')
def shutdown():
    clean_directory()
    print('Server Shutdown!')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/tts/')
async def tts_get_key(text: str, is_kor: str):
    print(f'get text:: {text} | is_kor::{is_kor}')
    key = run_synthesizer(text, bool(int(is_kor)))

    if key is not None:
        print(f'key: {key}, send /output/result/LJSpeech/{key}.wav')
        return {'key': key}
    else:
        raise HTTPException(status_code=500, detail='Synthesis Failed')


@app.get('/stt/')
async def stt_get_key(text: str):
    print(f'get text:: {text}')
    answer = chat(text)
    print(f'answer is {answer}.')
    key = run_synthesizer(answer, False)

    if key is not None:
        print(f'key: {key}, send /output/result/LJSpeech/{key}.wav')
        return {'key': key, 'text': answer}
    else:
        raise HTTPException(status_code=500, detail='Synthesis Failed')


@repeat_every(seconds=60*30) # 30 minutes
def clean_cache():
    from os import remove

    for key in cache.get_expired():
        path = f'./audios/{key}.wav'
        remove(path)
        cache.del_key(key)
