class Log:
    def debug(self, tag, msg):
        print(f"[DEBUG][{tag}] {msg}")

    def info(self, tag, msg):
        print(f"[INFO][{tag}] {msg}")

    def warn(self, tag, msg):
        print(f"[WARN][{tag}] {msg}")

    def error(self, tag, msg):
        print(f"[ERROR][{tag}] {msg}")

log = Log()
