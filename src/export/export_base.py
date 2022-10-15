from abc import abstractmethod
import io


class ExportBase:
    def __init__(self, writer: io.TextIOWrapper):
        self.fd = writer
        
    @abstractmethod
    def write(self, *args) -> None:
        """
        This method should convert the given predictions to the
        export format and write it to a file
        """
    
    @abstractmethod
    def finish(self) -> None:
        """
        This method should finish generating the output and flush
        it to the output file.
        """
    