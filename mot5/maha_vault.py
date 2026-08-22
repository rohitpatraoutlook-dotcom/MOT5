"""
Maha Vault — Append-Only Memory with GitHub Sync
"""
import json
import os
import requests
import base64
from datetime import datetime

class MahaVault:
    def __init__(self):
        self.relations = {}
        self.metadata = {
            "created": datetime.now().isoformat(),
            "total_relations": 0,
            "note": "Only operations stored, no numbers!"
        }
        self.token = os.environ.get("MOT5_TOKEN", "")
        self.repo = "rohitpatraoutlook-dotcom/MOT5"
        self.file_path = "maha_vault.json"
        self.branch = "main"
    
    def add_relation(self, pattern, operation_sequence):
        ops_only = [op if isinstance(op, str) else op[0] for op in operation_sequence]
        ops_str = "→".join(ops_only)
        
        if pattern not in self.relations:
            self.relations[pattern] = []
        
        if ops_str not in [self._to_ops_str(s) for s in self.relations[pattern]]:
            self.relations[pattern].append(ops_only)
            self.metadata["total_relations"] += 1
            return True
        return False
    
    def _to_ops_str(self, seq):
        return "→".join([op if isinstance(op, str) else op[0] for op in seq])
    
    def get_relations(self, pattern):
        return self.relations.get(pattern, [])
    
    def save(self, filepath=None):
        if filepath is None:
            filepath = "maha_vault.json"
        with open(filepath, 'w') as f:
            json.dump({
                'relations': self.relations,
                'metadata': self.metadata
            }, f, indent=2)
    
    def load(self, filepath=None):
        if filepath is None:
            filepath = "maha_vault.json"
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.relations = data.get('relations', {})
                self.metadata = data.get('metadata', self.metadata)
        except:
            pass
    
    # 🔥 GITHUB SYNC
    def pull_from_github(self):
        """Download latest vault from GitHub"""
        if not self.token:
            print("⚠️ MOT5_TOKEN not set! Cannot pull from GitHub.")
            return False
        
        try:
            url = f"https://api.github.com/repos/{self.repo}/contents/{self.file_path}"
            headers = {"Authorization": f"token {self.token}"}
            resp = requests.get(url, headers=headers)
            
            if resp.status_code == 200:
                content = resp.json().get('content', '')
                if content:
                    decoded = base64.b64decode(content).decode()
                    data = json.loads(decoded)
                    self.relations = data.get('relations', {})
                    self.metadata = data.get('metadata', self.metadata)
                    print(f"✅ Pulled {len(self.relations)} relations from GitHub")
                    return True
        except Exception as e:
            print(f"⚠️ Pull failed: {e}")
        return False
    
    def push_to_github(self):
        """Push vault to GitHub"""
        if not self.token:
            print("⚠️ MOT5_TOKEN not set! Cannot push to GitHub.")
            return False
        
        try:
            # Save local
            self.save("maha_vault.json")
            
            with open("maha_vault.json", "r") as f:
                content = f.read()
            
            url = f"https://api.github.com/repos/{self.repo}/contents/{self.file_path}"
            headers = {"Authorization": f"token {self.token}"}
            
            # Get current SHA
            resp = requests.get(url, headers=headers)
            sha = resp.json().get('sha') if resp.status_code == 200 else None
            
            # Upload
            payload = {
                "message": f"Auto-update Maha Vault ({datetime.now().isoformat()})",
                "content": base64.b64encode(content.encode()).decode(),
                "branch": self.branch
            }
            if sha:
                payload["sha"] = sha
            
            resp = requests.put(url, headers=headers, json=payload)
            if resp.status_code in [200, 201]:
                print(f"✅ Pushed {len(self.relations)} relations to GitHub")
                return True
        except Exception as e:
            print(f"⚠️ Push failed: {e}")
        return False
