import spacy

english = spacy.load("en_core_web_sm") #english library

def cleaning(text):
    text = english(text)
    tokens = [token.lemma_.lower() for token in text if token.is_alpha and not token.is_stop]
    text = ' '.join(tokens)
    return english(text)

text = 'NLP is a fascinating technique! I started learning it 2 days ago.'
text = cleaning(text)
print(text)
