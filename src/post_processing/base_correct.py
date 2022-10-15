from abc import abstractmethod


class BaseCorrection:
    @abstractmethod
    def get_candidates(self, word: str):
        """
        This method should return a list of possible candidates
        for the given word each with a score.

        Args:
            word (str): misspelled word to retrieve the candidates for
        """
        raise NotImplementedError('get_candidates is not implemented on the base correction class.')