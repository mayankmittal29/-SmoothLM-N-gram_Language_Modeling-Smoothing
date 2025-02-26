import re
from nltk.tokenize import word_tokenize, sent_tokenize
def tokenize_text(text):
    
        patterns = {
            # r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?' : '<URL>',
            r'#[\w$%&-]+': '<HASHTAG>',                    # Hashtags
            r'@\w+': '<MENTION>',                    # Mentions (e.g., @username)
            r'\d+\s*%': '<PERCENTAGE>',                 # Percentages (e.g., 10%)
            r'\d+\s*(?:years|yr|y)\s*old': '<AGE>',  # Age values (e.g., 25 years old)
            r'\d{1,2}[:/]\d{1,2}': '<TIME>',        # Time (e.g., 12:30 or 12/30)
            r'(?:\d{1,2}\s*(?:am|pm))': '<TIME>',   # Time in 12-hour format (e.g., 12:30pm)
            r'\d+\s*(?:months?|years?|days?)': '<TIME_PERIOD>' , # Time periods (e.g., 3 months)
            r'\d{10}': '<PHONE>',                        # Phone numbers (simple 10-digit matching, can be adjusted)
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': '<EMAIL>',  # Email addresses
            r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}?': '<DATE>',  # Format: 27th December 2007
            r'[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}?': '<DATE>',  # Format: December 27th, 2007
            # r'S.' : 'South',
            # r'N.' : 'North',
            # r'W.' : 'West',
            # r'E.' : 'East',
            r'\d+°' :'<DEGREE>',
            r'\w\s?\.\s?\w\s?\.\s?\w': '<ABBREVATION>',                 # Abbrevation
            r'[A-Za-z0-9_-]+\.(?:zip?|txt?|htm?)' : '<FILENAME>',     # Filename
            # r'viz.' :'<viz>',
            # r'e.g.' : '<eg>',
        }
        text = text.replace('''"''','')
        text = text.replace('...','')
        text = text.replace('--',' ')
        text = text.replace('viz.','viz')
        text = text.replace('e.g.','eg')
        text = re.sub(r'\b([A-Z])\.', r'\1', text)
        text = re.sub(r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sr|Jr|Capt|Col|Gen|Rev|Hon|Fr|St|Sir|Madam|Mx)\.\s*','',text)
        text = re.sub(r'\(\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\)', r'(<URL>)', text)
        text = re.sub(r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?', '<URL>', text)
        text = re.sub(r'\(([^()]*?)[!?.]([^()]*)\)', r'(\1\2)', text)
        # Replace all special patterns with placeholders
        for pattern, placeholder in patterns.items():
            text = re.sub(pattern, placeholder, text)
        # Adjust the sentence pattern to include punctuation
        sentence_pattern = r'([.!?;])'
        # Split the text into sentences and keep the punctuation with the sentence
        sentences = re.split(sentence_pattern, text)
        # Reorganize the split parts back into sentences including punctuation
        sentence_groups = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip() + sentences[i+1].strip()
            sentence_groups.append(sentence)
        if(len(sentences)%2==1):
            sentence_groups.append(sentences[-1].strip())
        # Tokenize each sentence and preserve punctuation
        tokenized_sentences = []
        for sentence in sentence_groups:
            # Match words, punctuation, and other significant symbols
            tokens = re.findall(r'[A-Za-z0-9<>’\'@#%\$\d\-]+(?:_[A-Za-z0-9]+)*|[.,!?;:"()\[\]]', sentence.strip())
            if tokens and tokens != ['.']:  # Only add non-empty sentences
                tokenized_sentences.append(tokens)

        return tokenized_sentences

if __name__ == '__main__':
    text = input("Your Input text: ")
    tokens = tokenize_text(text)
    print("Tokenized Text: ", tokens)
