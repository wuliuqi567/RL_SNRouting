import atexit
import sys

class Logger(object):
    def __init__(self, filename='logfile.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        atexit.register(self.close)  # Register the close method to be called when the program exits

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()