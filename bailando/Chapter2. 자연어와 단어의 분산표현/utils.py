import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# 당연히 자동화할 수 있습니다.
# corpus, vocab_size, window_size를 입력받아 행렬을 만드는 함수를 만들어봅니다.
def create_co_matrix(corpus: list, vocab_size: int, window_size=1):
    """Create co-occurrence matrix

    Args:
        corpus (list): 단어의 분산표현?
        vocab_size (int): 단어 사전 크기
        window_size (int, optional): 옆에 몇개까지 볼 것인지. Defaults to 1.

    Returns:
        np.array: 동시발생행렬
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

import numpy as np

def ppmi(C, verbose=False, eps=1e-8) -> list:
    """Make ppmi from 동시발생 행렬

    Args:
        C (list): 동시발생행렬
        verbose (bool, optional): 상세히 출력하기 위한 옵션. Defaults to False.
        eps (float, optional): log2가 음의 무한대가 되는 것을 막기 위한 임의의 작은 수. Defaults to 1e-8.

    Returns:
        list: ppmi
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print(f"{100*cnt/total:.1f} 완료")
    return M


# 책에서 하라고 한대로 따라하기
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 쿼리 확인
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    similarity = np.zeros(len(word_to_id))
    for i in range(len(word_to_id)):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순 정렬 및 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return