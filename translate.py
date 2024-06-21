import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
from tqdm import tqdm
import os
LANG = {
    'amh': 'amh_Ethi',
    'arq': 'ary_Arab',
    'ary': 'ary_Arab',
    'eng': 'eng_Latn',
    'hau': 'hau_Latn',
    'kin': 'kin_Latn',
    'mar': 'mar_Deva',
    'tel': 'tel_Telu'
}

def translate(data: pd.DataFrame, lang, mode):
    if lang == 'eng':
        data['Text'] = data['Text'].apply(lambda x: x.lower().strip().split('\n'))
        data['s0'] = data['Text'].apply(lambda x: x[0])
        data['s1'] = data['Text'].apply(lambda x: x[1])
        out = pd.DataFrame({
            's0': data['s0'], 
            's1': data['s1'],
            'Score': data["Score"]
        })
        if mode == 'train':
            out.to_csv(f'./nllb-1.3B-trans/{lang}/{lang}_train.csv')
        elif mode == 'dev':
            out.to_csv(f'./nllb-1.3B-trans/{lang}/{lang}_dev_with_labels.csv')
        elif mode == 'test':
            out.to_csv(f'./nllb-1.3B-trans/{lang}/{lang}_test_with_labels.csv')
        else:
            raise
        return

    tokenizer = AutoTokenizer.from_pretrained(
        '/home/maty/models/nllb-1.3B', token=True, scr_lang=LANG[lang]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained('/home/maty/models/nllb-1.3B')
    print(f'===begin {lang} {mode}===')
    data['Text'] = data['Text'].apply(lambda x: x.lower().strip().split('\n'))
    data['s0'] = data['Text'].apply(lambda x: x[0])
    data['s1'] = data['Text'].apply(lambda x: x[1])
    translated_0 = []
    translated_1 = []
    i = 0
    bsz = 64
    # 3592149
    while i < len(data):
        if i + bsz >= len(data):
            bsz = len(data)  - i
        inputs0 = tokenizer(data['s0'].iloc[i:i+bsz].tolist(), return_tensors='pt', padding=True, truncation=True)
        trans0 = model.generate(
            **inputs0, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'],
            max_length=75, repetition_penalty=1.3
        )
        trans0 = tokenizer.batch_decode(trans0, skip_special_tokens=True)
        inputs1 = tokenizer(data['s1'].iloc[i:i+bsz].tolist(), return_tensors='pt', padding=True, truncation=True)
        trans1 = model.generate(
            **inputs1, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'],
            max_length=75, repetition_penalty=1.3
        )
        trans1 = tokenizer.batch_decode(trans1, skip_special_tokens=True)
        translated_0 += trans0
        translated_1 += trans1
        i += bsz
    out = pd.DataFrame({
        's0': translated_0, 
        's1': translated_1,
        'Score': data["Score"]
    })
    if mode == 'train':
        out.to_csv(f'./nllb-1.3B-trans/{lang}/{lang}_train.csv')
    elif mode == 'dev':
        out.to_csv(f'./nllb-1.3B-trans/{lang}/{lang}_dev_with_labels.csv')
    elif mode == 'test':
        out.to_csv(f'./nllb-1.3B-trans/{lang}/{lang}_test_with_labels.csv')
    else:
        raise
    del model
    del tokenizer

def trans(sentence, lang):
    tokenizer = AutoTokenizer.from_pretrained(
        '/home/maty/models/nllb-1.3B', token=True, scr_lang=LANG[lang]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained('/home/maty/models/nllb-1.3B')
    inputs0 = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    trans0 = model.generate(
        **inputs0, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn']
    )
    trans0 = tokenizer.batch_decode(trans0, skip_special_tokens=True)
    print(trans0)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.system('mkdir nllb-1.3B-trans/')
    for lang in tqdm(LANG.keys()):
        os.system(f'mkdir ./nllb-1.3B-trans/{lang}')
        train_path = f"./Semantic_Relatedness_SemEval2024/Track A/{lang}/{lang}_train.csv"
        dev_path = f"./Semantic_Relatedness_SemEval2024/Track A/{lang}/{lang}_dev_with_labels.csv"
        test_path = f"./Semantic_Relatedness_SemEval2024/Track A/{lang}/{lang}_test_with_labels.csv"
        train_data = pd.read_csv(train_path, usecols=['Text', 'Score'])
        dev_data = pd.read_csv(dev_path, usecols=['Text', 'Score'])
        test_data = pd.read_csv(test_path, usecols=['Text', 'Score'])
        translate(train_data, lang, 'train')
        translate(dev_data, lang, 'dev')
        translate(test_data, lang, 'test')