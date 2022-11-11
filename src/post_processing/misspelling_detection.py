import os
import os.path as osp
import random

langConfig = {
    'en': osp.join(os.getcwd(), 'assets/dictionary_en.txt'),
    'de': osp.join(os.getcwd(), 'assets/frequency_dictionary_de.txt')
}

class MisspellingDetection:
    def __init__(self, lang:str='en') -> None:
        if lang not in langConfig:
            raise F"[ERROR] MisspellingDetection: Unsupported language {lang}"
        
        self.dictionary = set()
        self.charchterSet = set()
        
        # build dictionary of words
        with open(langConfig[lang], mode='r') as langFile:
            line = langFile.readline()
            while line:
                word = line.split(' ')[0].lower().strip()
                self.dictionary.add(word)
                self.charchterSet.update([*word])
                line = langFile.readline()
        
        print('[INFO] MisspellingDetection: random sample of words from dictionary : ', random.sample(self.dictionary, 10))
        
        print(f'[INFO] MisspellingDetection: dictionary built with {len(self.dictionary)} words.')
        print(f'[INFO] MisspellingDetection: available chrachter set: {self.charchterSet}')
        
    def check(self, word: str):
        return word.lower() in self.dictionary
        