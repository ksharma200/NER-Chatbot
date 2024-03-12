import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import time

class SimpleLabeler():
    def __init__(self, excel_path):
        self.t_nlp = spacy.blank('en')
        if 'ner' not in self.t_nlp.pipe_names:
            self.t_nlp.add_pipe('ner')
        self.t_nlp.begin_training()
        self.excel_df = pd.read_excel(excel_path, sheet_name='Label')
        self.matcher = Matcher(vocab=self.t_nlp.vocab)
        self.labels = list(self.excel_df.columns)
        self.load_stuff()

    def load_stuff(self):
        for label in self.excel_df.columns:
            for text in self.excel_df[label].dropna().unique():
                words = text.split()
                if len(words) > 1: 
                    pattern = [{'LOWER': word.lower()} for word in words]
                else:
                    pattern = [{'LOWER': words[0].lower()}]
                self.matcher.add(label.upper(), [pattern])

    def __call__(self, text):
        doc = self.t_nlp(text)
        matches = self.matcher(doc)
        spans = [] 
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=self.t_nlp.vocab.strings[match_id])
            spans.append(span)
        filtered_spans = spacy.util.filter_spans(spans)
        doc.ents = filtered_spans
        return doc
    
___all___ = "SimpleLabeler"
