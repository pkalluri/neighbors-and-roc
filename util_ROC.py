import csv

def get_stories_contexts_and_completions(ROC_filepath, num_stories=None):
    stories = []
    contexts = []
    completions = []
    with open(ROC_filepath, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader)  # Ignore headers
        if num_stories == None:
            for row in reader:
                stories.append(row[2:])  # Ignore metainfo
                contexts.append(row[2:-1])  # Ignore metainfo and last sentence
                completions.append(row[-1]) # Last sentence
        else:
            for i in range(num_stories):
                row = next(reader)
                stories.append(row[2:])  # Ignore metainfo
                contexts.append(row[2:-1])  # Ignore metainfo and last sentence
                completions.append(row[-1])
    return stories, contexts, completions

def get_all_sentences(ROC_filepath, num_stories=None):
    sentences = []
    with open(ROC_filepath, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader)  # Ignore headers
        if num_stories == None:
            for row in reader:
                sentences.extend(row[2:])  # Ignore metainfo
        else:
            for i in range(num_stories):
                row = next(reader)
                sentences.extend(row[2:])  # Ignore metainfo
    return sentences

# Lumps each doc into a single string
def lump_sentences_into_docs(docs):
    return [' '.join(doc) for doc in docs]
