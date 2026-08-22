"""
MOT5 Ultimate - Auto GitHub Sync
"""
import os, json, base64, requests
from .mot5 import MOT5 as BaseMOT5

class MOT5Ultimate(BaseMOT5):
    REPO = "rohitpatraoutlook-dotcom/MOT5"
    TOKEN = os.environ.get("MOT5_TOKEN", "")
    
    def __init__(self, config=None, sync=True):
        self.sync = sync
        if sync and self.TOKEN:
            self._pull_memory()
        super().__init__(config)
        if sync and os.path.exists("memory_vault.json"):
            self.engine.memory.load("memory_vault.json")
    
    def fit(self, X, y):
        result = super().fit(X, y)
        if self.sync and self.TOKEN:
            self.engine.memory.save("memory_vault.json")
            self._push_memory()
        return result
    
    def _pull_memory(self):
        try:
            url = f"https://raw.githubusercontent.com/{self.REPO}/main/memory_vault.json"
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                open("memory_vault.json", "w").write(r.text)
        except:
            pass
    
    def _push_memory(self):
        try:
            with open("memory_vault.json", "r") as f:
                content = f.read()
            url = f"https://api.github.com/repos/{self.REPO}/contents/memory_vault.json"
            headers = {"Authorization": f"token {self.TOKEN}"}
            r = requests.get(url, headers=headers)
            sha = r.json().get("sha") if r.status_code == 200 else None
            payload = {"message": "Auto-update", "content": base64.b64encode(content.encode()).decode(), "sha": sha}
            requests.put(url, headers=headers, json=payload)
        except:
            pass
