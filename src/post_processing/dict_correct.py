from .base_correct import BaseCorrection
from .misspelling_detection import langConfig
from symspellpy import SymSpell, Verbosity

class SymSpellCorrect(BaseCorrection):    
    def __init__(self, lang:str='en') -> None:
        super().__init__()
        
        if lang not in langConfig:
            raise F"[ERROR] SymSpellCorrect: Unsupported language {lang}"
        
        self.dictionary_path = langConfig[lang]
        self.sym_spell = SymSpell()
        self.sym_spell.load_dictionary(self.dictionary_path, 0, 1)
    
    def get_candidates(self, term: str, max_edit_dist=2):
        candidates = self.sym_spell.lookup(term, Verbosity.CLOSEST, max_edit_distance=max_edit_dist)
        return candidates