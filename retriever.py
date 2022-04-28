import json
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util

PASSAGES_FILE = 'data/dev_art_150_dict.json'
QUESTION_DICT = 'data/dev_question_dict.json'

# Choose the retriever stategy: SPARSE | DENSE | RERANK | DPR
STRATEGY = 'SPARSE'

# Model for reranking
ce_model = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

# Model for dense retriever
bi_encoder = SentenceTransformer('all-mpnet-base-v2')

# Model for DPR
passage_encoder = SentenceTransformer(
    'facebook-dpr-ctx_encoder-single-nq-base')
query_encoder = SentenceTransformer(
    'facebook-dpr-question_encoder-single-nq-base')


def dpr_retriever(question, corpus, k):
    passage_embeddings = passage_encoder.encode(corpus)
    query_embedding = query_encoder.encode(question)
    scores = util.cos_sim(query_embedding, passage_embeddings)
    best_ids = scores[0].numpy().argsort()[-k:][::-1]
    return [corpus[id] for id in best_ids]


def dense_retriever(question, corpus, k):
    passage_embeddings = bi_encoder.encode(corpus)
    query_embedding = bi_encoder.encode(question)
    scores = util.cos_sim(query_embedding, passage_embeddings)
    best_ids = scores[0].numpy().argsort()[-k:][::-1]
    return [corpus[id] for id in best_ids]


def tf_idf_retriever(question, corpus, k):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([question])
    results = util.dot_score(query_vec.toarray(), X.toarray())
    best_ids = results[0].numpy().argsort()[-k:][::-1]
    return [corpus[i]for i in best_ids]


def rerank_ce(question, paras):
    inp = [(question, x) for x in paras]
    scores = ce_model.predict(inp, batch_size=100)
    idxs = scores.argsort()[::-1]
    return [paras[i] for i in idxs]


def evaluate_retrieve(q_dict: dict, art_dict: dict, k: int):
    correct_count = 0
    least_1_count = 0
    for qid, q_info in tqdm(q_dict.items()):
        art_id = q_info['article_id']
        q = q_info['text']
        evidence = q_info['evidence']
        highlight_evidence = q_info['highlighted_evidence']
        answer = q_info['extractive_spans']
        paras = art_dict[art_id]
        if STRATEGY == 'DENSE':
            retrieve = dense_retriever(q, paras, k)
        elif STRATEGY == 'SPARSE':
            retrieve = tf_idf_retriever(q, paras, k)
        elif STRATEGY == 'RERANK':
            retrieve = rerank_ce(q, paras)[:k]
        elif STRATEGY == 'DPR':
            dpr_retriever(q, paras, k)
        else:
            raise ValueError('Unknown STRATEGY')

        for r in retrieve:
            if any([ans in r for ans in answer]):
                least_1_count += 1
                break
        all_retr = ' '.join(retrieve)
        if all([ans in all_retr for ans in answer]):
            correct_count += 1
    acc = round(correct_count/len(q_dict)*100, 2)
    least_1_acc = round(least_1_count/len(q_dict)*100, 2)
    return acc, least_1_acc


if __name__ == '__main__':
    with open(PASSAGES_FILE) as f:
        art_dict = json.load(f)

    with open(QUESTION_DICT) as f:
        q_dict = json.load(f)

    for i in [1, 3, 5, 10, 15, 20]:
        print('K=', i)
        acc = evaluate_retrieve(q_dict, art_dict, i)

    ks = [1, 3, 5, 10, 15, 20, 25]
    accs = [evaluate_retrieve(q_dict, art_dict, k) for k in ks]
    print('Accuracy (contains all answers):', [x[0] for x in accs])
    print('Accuracy (contains at least 1 answer):', [x[1] for x in accs])
