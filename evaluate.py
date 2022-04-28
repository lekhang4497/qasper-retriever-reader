import json
from tqdm import tqdm
from transformers import pipeline
import torch
from retriever import dpr_retriever, dense_retriever, tf_idf_retriever, rerank_ce

PASSAGES_FILE = 'data/dev_art_150_dict.json'
QUESTION_DICT = 'data/dev_question_dict.json'
QA_MODEL = ''

# Choose the retriever stategy: SPARSE | DENSE | RERANK | DPR
STRATEGY = 'SPARSE'
ORACLE = False

question_answerer = pipeline(
    "question-answering",
    model=QA_MODEL,
    device=0 if torch.cuda.is_available() else -1
)


def find_answer(q, paragraphs):
    if STRATEGY == 'DENSE':
        context = dense_retriever(q, paragraphs, 1)[0]
    elif STRATEGY == 'SPARSE':
        context = tf_idf_retriever(q, paragraphs, 1)[0]
    elif STRATEGY == 'RERANK':
        context = rerank_ce(q, paragraphs)[0]
    elif STRATEGY == 'DPR':
        context(q, paragraphs, 1)[0]
    else:
        raise ValueError('Unknown STRATEGY')
    answer_info = question_answerer(
        question=q,
        context=context,
        max_answer_len=500,
        max_seq_len=512,
        max_question_len=128,
    )
    return answer_info['answer']


def predict_qasper(q_dict: dict, art_dict: dict):
    pred = {}
    for qid, q_info in tqdm(q_dict.items()):
        art_id = q_info['article_id']
        q = q_info['text']
        paras = [p for p in art_dict[art_id] if len(p.strip()) > 0]
        if ORACLE:
            paras = q_info['evidence']
        answer = find_answer(q, paras)
        pred[qid] = answer
    return pred


if __name__ == '__main__':
    with open(PASSAGES_FILE) as f:
        art_dict = json.load(f)

    with open(QUESTION_DICT) as f:
        q_dict = json.load(f)

    pred = predict_qasper(q_dict, art_dict)

    convert_pred = [{'question_id': k, 'predicted_answer': v,
                    'predicted_evidence': []} for k, v in pred.items()]

    with open('predictions.json', 'w') as f:
        f.write('\n'.join([json.dumps(x, ensure_ascii=False)
                for x in convert_pred]))
