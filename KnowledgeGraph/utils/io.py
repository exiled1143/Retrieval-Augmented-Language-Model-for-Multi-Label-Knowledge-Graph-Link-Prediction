import os

def check_dir(path: str):
    roots = []
    if not os.path.exists(os.path.dirname(path)):
        check_path = path
        while True:
            if os.path.dirname(check_path) != "":
                root = os.path.dirname(check_path)
                roots.insert(0, root)
                check_path = root
            else:
                break
        
        for root in roots:
            if not os.path.exists(root):
                os.mkdir(root)


