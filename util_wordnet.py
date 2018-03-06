import re
import nltk
from nltk.corpus import wordnet as wn
nltk.data.path.append('./nltk_corpus/')

"""True iff WordNet has at least one definition of given word"""
def has_definition(word):
    synsets = wn.synsets(word)
    if len(synsets) > 0:
        return True
    else:
        return False

def get_synonym(word):
    for synset in wn.synsets(word):
        if synset.

"""Get definitional gender of given word
 by extracting gender from WordNet definitions"""
def definitional_gender(word):
    # Definitional gender pairs, from Bolukbasi et al.
    definitional_pairs = [["woman", "man"], ["women", "men"], ["girl", "boy"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"],
                          ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"], ["wife", "husband"]]
    female_markers = []
    male_markers = []
    for pair in definitional_pairs:
        female_markers.append(pair[0])
        male_markers.append(pair[1])

    if word in female_markers: return Gender.FEMALE
    if word in male_markers: return Gender.MALE

    found_female = False
    found_male = False
    # for synset in wn.synsets(word):
    synset = wn.synsets(word)[0]
    definition = synset.definition()
    definition_words = re.findall(r"[\w']+|[^\w\s]", definition.lower())
    found_female = found_female or any([marker in definition_words for marker in female_markers])
    found_male = found_male or any([marker in definition_words for marker in male_markers])

    if found_female and not found_male: return Gender.FEMALE
    if found_male and not found_female: return Gender.MALE
    if found_female and found_male: return Gender.BOTH
    else: return Gender.NEUTRAL
