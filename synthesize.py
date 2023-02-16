import re
import argparse
import math
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def writefile(body, fname):
    out = open(fname, 'w')
    for line in body:
        out.write('{}\n'.format(line))
    out.close()


def readRules(pver, rule_book):
    if pver == 2:
        f = open(rule_book, 'r')
    elif pver == 3:
        f = open(rule_book, 'r', encoding="utf-8")

    rule_in = []
    rule_out = []

    while True:
        line = f.readline()

        line = re.sub('\n', '', line)

        if line != u'':
            if line[0] != u'#':
                IOlist = line.split('\t')
                rule_in.append(IOlist[0])
                if IOlist[1]:
                    rule_out.append(IOlist[1])
                else:  # If output is empty (i.e. deletion rule)
                    rule_out.append(u'')
        if not line: break
    f.close()

    return rule_in, rule_out


def isHangul(charint):
    hangul_init = 44032
    hangul_fin = 55203
    return charint >= hangul_init and charint <= hangul_fin


def checkCharType(var_list):
    #  1: whitespace
    #  0: hangul
    # -1: non-hangul
    checked = []
    for i in range(len(var_list)):
        if var_list[i] == 32:  # whitespace
            checked.append(1)
        elif isHangul(var_list[i]):  # Hangul character
            checked.append(0)
        else:  # Non-hangul character
            checked.append(-1)
    return checked


def graph2phone(graphs):
    # Encode graphemes as utf8
    try:
        graphs = graphs.decode('utf8')
    except AttributeError:
        pass

    integers = []
    for i in range(len(graphs)):
        integers.append(ord(graphs[i]))

    # Romanization (according to Korean Spontaneous Speech corpus; 성인자유발화코퍼스)
    phones = ''
    ONS = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp',
           's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
    NUC = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
           'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
    COD = ['', 'kf', 'kk', 'ks', 'nf', 'nc', 'nh', 'tf',
           'll', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh',
           'mf', 'pf', 'ps', 's0', 'ss', 'oh', 'c0', 'ch',
           'kh', 'th', 'ph', 'h0']

    # Pronunciation
    idx = checkCharType(integers)
    iElement = 0
    while iElement < len(integers):
        if idx[iElement] == 0:  # not space characters
            base = 44032
            df = int(integers[iElement]) - base
            iONS = int(math.floor(df / 588)) + 1
            iNUC = int(math.floor((df % 588) / 28)) + 1
            iCOD = int((df % 588) % 28) + 1

            s1 = '-' + ONS[iONS - 1]  # onset
            s2 = NUC[iNUC - 1]  # nucleus

            if COD[iCOD - 1]:  # coda
                s3 = COD[iCOD - 1]
            else:
                s3 = ''
            tmp = s1 + s2 + s3
            phones = phones + tmp

        elif idx[iElement] == 1:  # space character
            tmp = '#'
            phones = phones + tmp

        phones = re.sub('-(oh)', '-', phones)
        iElement += 1
        tmp = ''

    # 초성 이응 삭제
    phones = re.sub('^oh', '', phones)
    phones = re.sub('-(oh)', '', phones)

    # 받침 이응 'ng'으로 처리 (Velar nasal in coda position)
    phones = re.sub('oh-', 'ng-', phones)
    phones = re.sub('oh([# ]|$)', 'ng', phones)

    # Remove all characters except Hangul and syllable delimiter (hyphen; '-')
    phones = re.sub('(\W+)\-', '\\1', phones)
    phones = re.sub('\W+$', '', phones)
    phones = re.sub('^\-', '', phones)
    return phones


def phone2prono(phones, rule_in, rule_out):
    # Apply g2p rules
    for pattern, replacement in zip(rule_in, rule_out):
        # print pattern
        phones = re.sub(pattern, replacement, phones)
        prono = phones
    return prono


def addPhoneBoundary(phones):
    # Add a comma (,) after every second alphabets to mark phone boundaries
    ipos = 0
    newphones = ''
    while ipos + 2 <= len(phones):
        if phones[ipos] == u'-':
            newphones = newphones + phones[ipos]
            ipos += 1
        elif phones[ipos] == u' ':
            ipos += 1
        elif phones[ipos] == u'#':
            newphones = newphones + phones[ipos]
            ipos += 1

        newphones = newphones + phones[ipos] + phones[ipos + 1] + u','
        ipos += 2

    return newphones


def addSpace(phones):
    ipos = 0
    newphones = ''
    while ipos < len(phones):
        if ipos == 0:
            newphones = newphones + phones[ipos] + phones[ipos + 1]
        else:
            newphones = newphones + ' ' + phones[ipos] + phones[ipos + 1]
        ipos += 2

    return newphones


def graph2prono(graphs, rule_in, rule_out):
    romanized = graph2phone(graphs)
    romanized_bd = addPhoneBoundary(romanized)
    prono = phone2prono(romanized_bd, rule_in, rule_out)

    prono = re.sub(u',', u' ', prono)
    prono = re.sub(u' $', u'', prono)
    prono = re.sub(u'#', u'-', prono)
    prono = re.sub(u'-+', u'-', prono)

    prono_prev = prono
    identical = False
    loop_cnt = 1

    while not identical:
        prono_new = phone2prono(re.sub(u' ', u',', prono_prev + u','), rule_in, rule_out)
        prono_new = re.sub(u',', u' ', prono_new)
        prono_new = re.sub(u' $', u'', prono_new)

        if re.sub(u'-', u'', prono_prev) == re.sub(u'-', u'', prono_new):
            identical = True
            prono_new = re.sub(u'-', u'', prono_new)
        else:
            loop_cnt += 1
            prono_prev = prono_new

    return prono_new

def runKoG2P(graph, rulebook):
    [rule_in, rule_out] = readRules(3, rulebook)
    prono = graph2prono(graph, rule_in, rule_out)

    return prono

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config, is_kor=False):
    print(f'{text} | {preprocess_config} | {is_kor}')
    print('31')
    if not is_kor:
        print('32')
        text = text.rstrip(punctuation)
        lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
        
        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))

        print(phones)
        phones = "{" + "}{".join(phones) + "}"
        print(phones)
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        print(phones)
        phones = phones.replace("}{", " ")
        print(phones)
    else:
        print('32')

        print('33')
        print('hi')
        print(f'{text}')

        table = {'p0': 'B', 'ph': 'B', 'pp': 'F', 't0': 'D', 'th': 'S', 'tt': 'DH', 'k0': 'G', 'kh': 'G', 'kk': 'G', 's0': 'S', 'ss': 'S', 'h0': 'HH', 'c0': 'Z', 'ch': 'S', 'cc': 'ZH', 'mm': 'M', 'nn': 'N', 'rr': 'R', 'pf': 'B', 'tf': 'D', 'kf': 'G', 'mf': 'M', 'nf': 'N', 'ng': 'NG', 'll': 'L', 'ks': 'G', 'nc': 'N', 'nh': 'N', 'lk': 'G', 'lm': 'L', 'lb': 'L', 'ls': 'L', 'lt': 'L', 'lp': 'L', 'lh': 'L', 'ps': 'B', 'ii': 'IH0', 'ee': 'AE0', 'qq': 'AE0', 'aa': 'AH0', 'xx': 'UH0', 'vv': 'EH0', 'uu': 'UW', 'oo': 'OW0', 'ye': 'Y AE0', 'yq': 'Y AE0', 'ya': 'Y AH0', 'yv': 'IH0 EH0', 'yu': 'Y UW', 'yo': 'Y OW0', 'wi': 'W IH0', 'wo': 'W AE0', 'wq': 'W AE0', 'we': 'W AE0', 'wa': 'W AH0', 'wv': 'W EH0', 'xi': 'W IH0'}
        result = runKoG2P(text, './KoG2P/rulebook.txt')

        print('44')
        print(result)

        phones = []
        for pronun in result.split(' '):
            phones.append(table[pronun])
        print(phones)
        print(f'kor: {phones}')
        phones = "{" + "}{".join(phones) + "}"
        print(f'kor: {phones}')
        phones = phones.replace("}{", " ")
        print(f'kor: {phones}')
    

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    print(np.array(sequence))

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values, file_name=None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                file_name=file_name
            )


def run_synthesizer(raw_text: str, is_kor=False):
    from argparse import Namespace
    from server.random_text import get_random_key

    args = Namespace(restore_step=900000, mode='single', source=None, text=raw_text, speaker_id=0,
                     preprocess_config='config/LJSpeech/preprocess.yaml', model_config='config/LJSpeech/model.yaml',
                     train_config='config/LJSpeech/train.yaml', pitch_control=1.0, energy_control=1.0,
                     duration_control=1.0, is_korean=0)

    try:
        # Read Config
        preprocess_config = yaml.load(
            open(args.preprocess_config, "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
        configs = (preprocess_config, model_config, train_config)

        model = get_model(args, configs, device, train=False)

        vocoder = get_vocoder(model_config, device)

        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        texts = np.array([preprocess_english(args.text, preprocess_config, is_kor)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

        control_values = args.pitch_control, args.energy_control, args.duration_control


        file_name = get_random_key()
        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, file_name=file_name)

        return file_name
    except Exception as e:
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--is_korean",
        type=int,
        default=0,
        help="Korean test",
    )
    args = parser.parse_args()

    from argparse import Namespace
    args = Namespace(restore_step=900000, mode='single', source=None, text='Hello world', speaker_id=0, preprocess_config='config/LJSpeech/preprocess.yaml', model_config='config/LJSpeech/model.yaml', train_config='config/LJSpeech/train.yaml', pitch_control=1.0, energy_control=1.0, duration_control=1.0, is_korean=0)
    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

    print('11')
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            print('22')
            print(f'test {args.text}')
            texts = np.array([preprocess_english(args.text, preprocess_config, bool(args.is_korean))])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
