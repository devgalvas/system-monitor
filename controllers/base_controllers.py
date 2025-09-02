from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, dataloader, namespace=None):
        self.dl = dataloader
        
        if namespace != None:
            self.namespace = namespace
            if isinstance(namespace, list):
                self.view_name = [f"{ns.replace('-', '_')}_view" for ns in namespace]
            else:
                self.view_name = f"{self.namespace.replace('-', '_')}_view"


    @abstractmethod
    def run():
        pass