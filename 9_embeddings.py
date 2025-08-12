import torch
import numpy as np


def load_word2vec_binary(file_path):
    """
    Loads Word2Vec binary format into PyTorch tensors.
    Assumes the file is in the original word2vec binary format.
    """
    with open(file_path, 'rb') as f:
        # Read header: vocab_size embed_dim
        header = f.readline().decode('utf-8').strip()
        vocab_size, embed_dim = map(int, header.split())

        words = []
        vectors_list = []
        binary_len = np.dtype(np.float32).itemsize * embed_dim

        for _ in range(vocab_size):
            # Read word (bytes until space)
            word_bytes = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':  # Ignore unexpected newlines
                    word_bytes.append(ch)
            word = b''.join(word_bytes).decode('utf-8')

            # Read vector as float32 binary
            vector = np.frombuffer(f.read(binary_len), dtype=np.float32)

            words.append(word)
            vectors_list.append(torch.from_numpy(vector))

    # Stack into a single tensor for efficient computation
    vectors = torch.stack(vectors_list)
    word_to_ix = {word: i for i, word in enumerate(words)}

    return words, vectors, word_to_ix, embed_dim


def most_similar(positive, negative, topn=10, restrict_vocab=None):
    """
    Finds topn most similar words using cosine similarity after vector arithmetic.
    Analogy: sum(positive) - sum(negative).
    """
    # Get indices for input words (skip missing ones)
    pos_ix = [word_to_ix.get(w) for w in positive if w in word_to_ix]
    neg_ix = [word_to_ix.get(w) for w in negative if w in word_to_ix]

    if not pos_ix:
        raise ValueError("No positive words found in vocabulary.")

    # Compute query vector
    pos_vecs = vectors[pos_ix]
    neg_vecs = vectors[neg_ix] if neg_ix else torch.zeros((0, embed_dim))
    query = torch.mean(pos_vecs, dim=0) - torch.mean(neg_vecs, dim=0)
    query = query / torch.norm(query)

    # Normalize all vectors
    norms = torch.norm(vectors, dim=1, keepdim=True)
    normalized_vectors = vectors / norms

    # Cosine similarities (matrix multiplication)
    sims = torch.matmul(normalized_vectors, query)

    # Exclude input words from results
    exclude_ix = set(pos_ix + neg_ix)
    for ix in exclude_ix:
        sims[ix] = -float('inf')

    # Optional: restrict to top restrict_vocab words (for efficiency, as in Gensim)
    if restrict_vocab:
        sims[restrict_vocab:] = -float('inf')

    # Get topn indices and scores
    top_scores, top_ix = torch.topk(sims, topn)
    results = [(words[i], top_scores[j].item()) for j, i in enumerate(top_ix)]

    return results


# Load the model (replace with your file path)
file_path = 'GoogleNews-vectors-negative300.bin'  # After downloading and unzipping
words, vectors, word_to_ix, embed_dim = load_word2vec_binary(file_path)

# Perform the similarity query
result = most_similar(positive=['Xbox', 'Sony'], negative=['Microsoft'], topn=10)
print(result)