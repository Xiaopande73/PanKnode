from KBManager import KBManager
from GenerateManager import GenerateManager
from EmbeddingManager import EmbeddingManager

class PanKnode(GenerateManager, KBManager, EmbeddingManager):
    """
    Main project class for PanKnode.
    Inherits from various managers to provide a unified interface.
    """
    def __init__(self, db_path: str, kb_path: str, gen_path: str):
        # Initialize parent classes
        KBManager.__init__(self)
        GenerateManager.__init__(self)
        EmbeddingManager.__init__(self, db_path)

        self.db_path = db_path
        self.kb_path = kb_path
        self.gen_path = gen_path
