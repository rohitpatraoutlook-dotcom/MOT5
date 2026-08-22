"""
MOT5 Ultimate - Auto GitHub Sync
"""
import os, sys, json, base64, requests
# Import base MOT5 from parent
from mot5 import MOT5 as BaseMOT5

class MOT5(BaseMOT5):
    REPO = "rohitpatraoutlook-dotcom/MOT5"
    TOKEN = os.environ.get("MOT5_TOKEN", "")
    FILE_PATH = "memory_vault.json"
    BRANCH = "main"
    
    def __init__(self, config=None):
        if not self.TOKEN:
            print("⚠️ MOT5_TOKEN not set! Memory will not sync.")
        self._pull_memory()
        super().__init__(config)
        if os.path.exists("memory_vault.json"):
            self.engine.memory.load("memory_vault.json")
        print("🧠 MOT5 Ultimate Ready!", end="\r")
    
    def fit(self, X, y):
        result = super().fit(X, y)
        self.engine.memory.save("memory_vault.json")
        self._push_memory()
        return result
    
    def _pull_memory(self):
        try:
            url = f"https://raw.githubusercontent.com/{self.REPO}/{self.BRANCH}/{self.FILE_PATH}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                with open("memory_vault.json", "w") as f:
                    f.write(resp.text)
        except:
            self._create_fresh_memory()
    
    def _push_memory(self):
        if not self.TOKEN:
            return
        try:
            with open("memory_vault.json", "r") as f:
                content = f.read()
            url = f"https://api.github.com/repos/{self.REPO}/contents/{self.FILE_PATH}"
            headers = {"Authorization": f"token {self.TOKEN}"}
            resp = requests.get(url, headers=headers)
            sha = resp.json().get("sha") if resp.status_code == 200 else None
            payload = {
                "message": "Auto-update memory",
                "content": base64.b64encode(content.encode()).decode(),
                "sha": sha,
                "branch": self.BRANCH
            }
            requests.put(url, headers=headers, json=payload)
        except:
            pass
    
    def _create_fresh_memory(self):
        fresh = {
            "invariant_store": {},
            "gene_library": {},
            "transition_rules": {},
            "metadata": {"created": "2026-08-22", "version": "1.0.0", "total_runs": 0}
        }
        with open("memory_vault.json", "w") as f:
            json.dump(fresh, f, indent=2)
