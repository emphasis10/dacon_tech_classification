import re

def preprocessor(sentence):
    filter_char = ['○', '√', '&amp;', '*', '#', '/',
                    'ㅇ', '-', '<br>', '&40', '&41', ';', 
                    '？', '▲', '●', 'o', '?', '⦁', '◦', 
                    '□', '◎', '&gt', '&lt', '◆', '|', '\n']

    pattern = ["\([^)]*\)", "\<[^)]*\>", "\[[^)]*\]", "[0-9]+."]

    for i in filter_char:
        sentence = sentence.replace(i, '')

    for pat in pattern:
        sentence = re.sub(pat, '', sentence)

    return sentence