import os
from abc import ABC

class BaseView(ABC):
    def __init__(self, dir_path: str = "outputs/"):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)