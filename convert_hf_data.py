import json
from typing import List
from tqdm import tqdm

INP = 'qasper-train-v0.3.json'
OUT = 'hf_train.json'

with open(INP) as f:
    data = json.load(f)


def build_examples(question_id, question, raw_answers: List[str], evidence: List[str]):
    examples = []
    for evi in evidence:
        context = evi
        title = ''
        ans_start = []
        ans_text = []
        for ans in raw_answers:
            if ans in context:
                ans_start.append(context.index(ans))
                ans_text.append(ans)
        if ans_text:
            examples.append({
                "answers": {
                    "answer_start": ans_start,
                    "text": ans_text
                },
                "context": context,
                "id": f'{question_id}_{len(examples)}',
                "question": question,
                "title": title,

            })
    return examples


def convert_data(data):
    hf_data = []
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            question = qa_info["question"]
            # references = []
            for i, annotation_info in enumerate(qa_info["answers"]):
                answer_info = annotation_info["answer"]
                raw_answers = answer_info["extractive_spans"]
                if not raw_answers:
                    continue
                evidence = [text for text in answer_info["evidence"]
                            if "FLOAT SELECTED" not in text]
                hf_data.extend(build_examples(
                    f'{question_id}_{i}', question, raw_answers, evidence))
    return hf_data


hf_data = convert_data(data)

ids = [d['id'] for d in hf_data]
assert len(ids) == len(set(ids))

with open(OUT, 'w') as f:
    f.write('\n'.join([json.dumps(x, ensure_ascii=False) for x in hf_data]))
