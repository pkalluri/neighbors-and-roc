import random

### SIMPLE FUNCTIONS ###

def simple_identity_data(vocab_size, sequence_length, num_samples, stop_token):
    tokens = list(range(vocab_size))
    in_seqs = [[str(random.choice(tokens)) for token_to_gen in range(sequence_length)]
                    for sentence_to_gen in range(num_samples)]
    out_seqs = [s for s in in_seqs]

    in_sentences = [' '.join(seq) for seq in in_seqs]
    out_sentences = [' '.join(seq+[stop_token]) for seq in out_seqs]
    return in_sentences, out_sentences

def simple_reverse_data(vocab_size, sequence_length, num_samples, stop_token):
    tokens = list(range(vocab_size))
    in_seqs = [[str(random.choice(tokens)) for token_to_gen in range(sequence_length)]
                    for sentence_to_gen in range(num_samples)]
    out_seqs = [seq[::-1] for seq in in_seqs] # reverse

    in_sentences = [' '.join(seq) for seq in in_seqs]
    out_sentences = [' '.join(seq+[stop_token]) for seq in out_seqs]
    return in_sentences, out_sentences

def simple_remove_stops(vocab_size, sequence_length, num_samples, stop_token, prob_of_stops=.3):
    tokens = list(range(vocab_size))
    in_seqs = []
    in_seqs = [[str(random.choice(tokens)) if random.random()>prob_of_stops else str(stop_token) for token_to_gen in range(sequence_length)]
                    for sentence_to_gen in range(num_samples)]
    out_seqs = []
    for in_seq in in_seqs:
        out_seq = [token for token in in_seq if token!=stop_token]
        out_seqs.append(out_seq)

    in_sentences = [' '.join(seq) for seq in in_seqs]
    out_sentences = [' '.join(seq+[stop_token]) for seq in out_seqs]
    return in_sentences, out_sentences

### SIMPLE FUNCTIONS WITH PARAPHRASES ###

def identity_data(vocab_size, sequence_length, num_base_samples, percentage_samples_to_add, stop_token, num_synonyms, additional_data_is_alternative_data):
    if additional_data_is_alternative_data: return identity_data_with_alternatives(vocab_size, sequence_length,
                                                                                   num_base_samples,
                                                                                   percentage_samples_to_add,
                                                                                   stop_token, num_synonyms)
    else: return identity_data_unique(vocab_size, sequence_length, num_base_samples, percentage_samples_to_add,
                                                                                   stop_token, num_synonyms)

def identity_data_unique(vocab_size, sequence_length, num_base_samples, percentage_samples_to_add, stop_token, num_synonyms):
    tokens = list(range(vocab_size))
    in_seqs = []
    num_samples = num_base_samples + int(percentage_samples_to_add*num_base_samples)
    for _ in range(num_samples):
        added_unique = False
        while added_unique == False:
            in_seq = [str(random.choice(tokens)) for token_to_gen in range(sequence_length)]
            if not in_seq in in_seqs:
                in_seqs.append(in_seq)# add
                added_unique = True
    in_seqs_with_paraphrases = [[get_synonym(token, num_synonyms) for token in seq] for seq in in_seqs]
    out_seqs_with_paraphrases = [[get_synonym(token, num_synonyms) for token in seq] for seq in in_seqs]

    in_sentences = [' '.join(seq) for seq in in_seqs_with_paraphrases]
    out_sentences = [' '.join(seq+[stop_token]) for seq in out_seqs_with_paraphrases]
    return in_sentences, out_sentences

def identity_data_with_alternatives(vocab_size, sequence_length, num_base_samples, percentage_samples_to_add, stop_token, num_synonyms):
    tokens = list(range(vocab_size))
    in_seqs = []
    for _ in range(num_base_samples):
        added_unique = False
        while not added_unique:
            in_seq = [str(random.choice(tokens)) for token_to_gen in range(sequence_length)]
            if not in_seq in in_seqs:
                in_seqs.append(in_seq)
                added_unique = True
    in_seqs_with_paraphrases = [[get_synonym(token, num_synonyms) for token in seq] for seq in in_seqs]
    out_seqs_with_paraphrases = [[get_synonym(token, num_synonyms) for token in seq] for seq in in_seqs]

    samples = list(zip(in_seqs_with_paraphrases, out_seqs_with_paraphrases))
    in_seqs_to_add = []
    out_seqs_to_add = []
    num_samples_to_add = int(percentage_samples_to_add*num_base_samples)
    for _ in range(num_samples_to_add):
        (x,y) = random.choice(samples)
        in_seqs_to_add.append(x) #existing x
        added_alternative = False
        while not added_alternative:
            y_alternative = [get_synonym(token, num_synonyms) for token in x]
            if y_alternative != y: #alternative
                out_seqs_to_add.append(y_alternative)
                added_alternative = True
    in_sentences = [' '.join(seq) for seq in in_seqs_with_paraphrases+in_seqs_to_add]
    out_sentences = [' '.join(seq+[stop_token]) for seq in out_seqs_with_paraphrases+out_seqs_to_add]
    return in_sentences, out_sentences

SYNONYM_MARKER = '*'

def get_synonym(token, num_synonyms):
    return token.split(SYNONYM_MARKER)[0]+SYNONYM_MARKER*random.randint(0,num_synonyms) #randint is inclusive

def is_paraphrase(s1_tokens,s2_tokens):
    s1_tokens = [token.split(SYNONYM_MARKER)[0] for token in s1_tokens]
    s2_tokens = [token.split(SYNONYM_MARKER)[0] for token in s2_tokens]
    return s1_tokens==s2_tokens
