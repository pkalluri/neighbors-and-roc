import numpy as np

def get_embedding_matrix(embedding_size, embedding_path, vocab_size, word_to_index):
    embedding_matrix = np.random.rand(vocab_size, embedding_size)
    with open(embedding_path, "rb") as embedding_f:
        first_line = embedding_f.readline() #Skip

        words_found = 0
        for line in embedding_f:
            if words_found == vocab_size: break
            values = line.split()
            word = values[0]
            if word in word_to_index:
                coefs = np.asarray(values[1:],dtype='float32')
                embedding_matrix[word_to_index[word]] = coefs
                words_found += 1
    return embedding_matrix
#
# def get_vocabularys_embedding_matrix(embedding_size, embedding_path, vocabulary, topn=0):
#     # Get all embeddings
#     embeddings = {}
#     all_embeddings_words_by_frequency = []
#
#     # Get all embeddings
#     with open(embedding_path, "rb") as embedding_f:
#         first_line = embedding_f.readline()
#         # values = first_line.split()
#         # voca = int(values[0])
#         # d = int(values[1])
#
#         for line in embedding_f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:],dtype='float32')
#             all_embeddings_words_by_frequency.append(word)
#             embeddings[word] = coefs
#
#     # Build embedding matrix
#     num_vocabulary_words = vocabulary.get_size()
#     num_words_total = min(num_vocabulary_words + topn, vocabulary.get_size())
#     embedding_matrix = np.random.rand(num_words_total, embedding_size)
#     # Add in given vocabulary
#     for word in vocabulary:
#         embedding = embeddings.get(word)
#         if embedding is not None:
#             id = vocabulary.get_id(word)
#             embedding_matrix[id] = embedding
#     # Add given number of additional top words
#     num_words_to_add = num_words_total - num_vocabulary_words
#     num_words_added = 0
#     for word in all_embeddings_words_by_frequency:
#         if num_words_added==num_words_to_add:
#             break
#         if not vocabulary.contains(word):
#             id = vocabulary.add(word) # Add to vocabulary
#             embedding_matrix[id] = embeddings[word] # Add to embedding matrix
#             num_words_added += 1
#     # Done adding words
#     return embedding_matrix, vocabulary