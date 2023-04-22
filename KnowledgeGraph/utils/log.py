
class Log:
    show_print = False
    show_info = False
    show_debug = False
    show_ddp_info = False

    @classmethod
    def print(cls, *args, **kwargs):
        if cls.show_print:
            print(*args, **kwargs)

    @classmethod
    def info(cls, *args, **kwargs):
        if cls.show_info:
            print(*args, **kwargs)

    @classmethod
    def ddp_info(cls, *args, **kwargs):
        if cls.show_ddp_info:
            print(*args, **kwargs)
            
    @classmethod
    def debug(cls, *args, **kwargs):
        if cls.show_debug:
            print(*args, **kwargs)


