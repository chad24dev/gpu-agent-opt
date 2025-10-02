# gpu_agent_opt/knowledge.py
import json
import os

class KnowledgeBase:
    """
    Stores kernel configs and scores for reuse.
    """
    def __init__(self, gpu_name="default", save_path="kb.json"):
        self.gpu_name = gpu_name
        self.save_path = save_path
        self.db = {}
        if os.path.exists(save_path):
            try:
                with open(save_path, "r") as f:
                    self.db = json.load(f)
            except Exception:
                self.db = {}

    def store(self, config, score):
        key = str(config)
        self.db[key] = score
        with open(self.save_path, "w") as f:
            json.dump(self.db, f, indent=2)

    def get_best(self):
        if not self.db:
            return None, None
        best_config = max(self.db, key=lambda k: self.db[k])
        return eval(best_config), self.db[best_config]
