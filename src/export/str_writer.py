import io
from .export_base import ExportBase

class SubtitleWriter(ExportBase):
    def __init__(self, writer: io.TextIOWrapper, fps: float):
        super().__init__(writer=writer)
        self.fps = fps
        self.spf = 1/fps * 1000 # seconds per frame
        self.subtitleNum = 1
        self.lastSubtitleText = None
        self.lastSubtitleStartTime = 0
    
    def __formatMillis(self, millis):
        mill = int(millis % 1000)
        s=int((millis/1000)%60)
        m=int((millis/(1000*60))%60)
        h=int(millis/(1000*60*60))
        return f'{h:0>2}:{m:0>2}:{s:0>2},{mill:0>3}'
    
    def __write_subtitle(self, subtitleText, start, end):
        startTime = self.__formatMillis(start)
        endTime = self.__formatMillis(end)
        res = f'{self.subtitleNum}\n'
        res += f'{startTime} --> {endTime}\n'
        res += subtitleText if subtitleText else ''
        res += '\n\n'
        self.fd.write(res)
    
    def write(self, resultText, frameNum):
        subtitleText = ' '.join(resultText)
        if self.lastSubtitleText is not None and subtitleText != self.lastSubtitleText:
            # write old subtitle to file
            self.__write_subtitle(self.lastSubtitleText, self.lastSubtitleStartTime, self.spf * frameNum)
            # start a new subtitle
            self.subtitleNum += 1
            self.lastSubtitleText = subtitleText
            self.lastSubtitleStartTime = self.spf * frameNum
        else:
            self.lastSubtitleText = subtitleText
    
    def finish(self):
        self.__write_subtitle(self.lastSubtitleText, self.lastSubtitleStartTime, self.lastSubtitleStartTime * 100)