import re, textblob, language_tool_python

def score_wordcount(text, target_count) -> float:
    word_count = len(text.split())
    error = abs(target_count - word_count)/target_count
    return round((1 - error)*100, 5) # score

def score_avg_sentence_length(text) -> float: # between 12 and 20 is the ideal average sentence length
    sentences = re.split(r'(?<=[.!?]) +', text)  # split at punctuation + space
    sentences = [s.strip() for s in sentences if s.strip()]  # clean empty

    word_counts = [len(sentence.split()) for sentence in sentences if sentence]
    
    if not word_counts:
        return 0
    
    avg_sentence_length = sum(word_counts) / len(word_counts)

    if 12 < avg_sentence_length < 20:
        return 100
    
    else:
        error = min(abs(12-avg_sentence_length), abs(avg_sentence_length-20))
        return round((1 - (error/100))**2 * 100, 5)

def score_repetitiveness(text) -> float: # a vocabulary should have distinct words, in order to be appealing. Too many distinct words, however, can overcomplicate it
    pass
    word_count  = len(text.split())
    unique_word_count = len(set(text.split()))

    ratio  = unique_word_count/word_count
    if .5 <= ratio <= .7:
        return 100
    
    else:
        error = min(abs(.5-ratio), abs(ratio-.7))
        return round((1 - (error/100))**2 * 100, 5)

def score_neutrality(text) -> float: # while generating informational text, the LLM should maintain a neutral tone, so as not to become biased towards any element, or lose its objectivity
    sentiment_score = abs(textblob.TextBlob(text).sentiment.polarity)
    return round((1 - sentiment_score)**2 * 100, 5)

def score_grammar(text) -> float: # Being grammatically correct is crucial for any LLM; one the fundamental requirements.
    
    tool = language_tool_python.LanguageTool('en-US')
    errors = len(tool.check(text))
    word_count = len(text.split())

    error_ratio = errors/word_count

    return round((1 - error_ratio)**3 * 100, 5)

def scoreModel(text, target_word_count) -> float: # The main function that evaluates the effectiveness of a model by testing the quality of the text generated from it
    total = (score_wordcount(text, target_word_count) + 
             score_avg_sentence_length(text) + 
             score_repetitiveness(text) + 
             score_neutrality(text) +
             score_grammar(text)
    )

    return round(total/5, 3)