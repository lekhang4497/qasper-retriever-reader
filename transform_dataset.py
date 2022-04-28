import json
from tqdm import tqdm

INPUT_FILE = "qasper-dev-v0.3.json"
OUTPUT_FILE = "dev_question_dict.json"


def split_on_window(sequence, limit, step):
    ret = []
    split_sequence = sequence.split()
    l, r = 0, limit
    while r < len(split_sequence):
        s = " ".join(split_sequence[l:r])
        ret.append(s)
        l += step
        r += step
    if l < len(split_sequence):
        s = " ".join(split_sequence[l:r])
        ret.append(s)
    return ret


def to_paragraphs(data):
    id2paragraphs = {}
    for k, v in data.items():
        para = []
        para.append(v['abstract'])
        for section in v['full_text']:
            para.extend(section['paragraphs'])
        id2paragraphs[k] = para

    with open('dev_art_dict.json', 'w') as f:
        json.dump(id2paragraphs, f, indent=4, ensure_ascii=False)


def to_questions(data):
    id2question = {}
    for k, v in data.items():
        for qa in v['qas']:
            answerable = False
            evidence_set = set()
            for a in qa['answers']:
                if a['answer']['extractive_spans']:
                    answerable = True
                    evidence_set.update(a['answer']['evidence'])
            if answerable:
                id2question[qa['question_id']] = {
                    'article_id': k,
                    'text': qa['question'],
                    'evidence': list(evidence_set)
                }

    with open('dev_question_dict.json', 'w') as f:
        json.dump(id2question, f, indent=4, ensure_ascii=False)
