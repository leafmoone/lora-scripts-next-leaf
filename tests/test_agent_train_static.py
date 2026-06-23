import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]



class AgentTrainStaticTests(unittest.TestCase):
    def test_agent_train_page_contains_required_api_wiring(self):
        html = (ROOT / "frontend" / "dist" / "agent-train.html").read_text(encoding="utf-8")

        self.assertIn("/api/agent-train/sessions", html)
        self.assertIn("/api/agent-train/chat", html)
        self.assertIn("/api/agent-train/approve", html)
        self.assertIn("/api/agent-train/status/", html)
        self.assertIn("api_key", html)
        self.assertIn("function updateModelDebug", html)
        self.assertIn("API Key: 已填写", html)
        self.assertIn("api_key: document.getElementById('apiKey').value.trim()", html)
        self.assertIn("localStorage.setItem", html)
        save_local = html.split("function saveLocal()", 1)[1].split("function loadLocal()", 1)[0]
        self.assertNotIn("api_key", save_local)
        self.assertNotIn("apiKey", save_local)
        self.assertIn("const activePlan = plan || {}", html)
        self.assertIn("side_effect_level", html)
        self.assertIn("风险级别", html)
        self.assertIn("关键参数", html)
        self.assertIn("function renderKeyFields", html)
        self.assertIn("function syncKeyFieldToJson", html)
        self.assertIn("'pretrained_model_name_or_path', '底模'", html)
        self.assertIn("'learning_rate', '学习率'", html)
        self.assertIn("missing_slots", html)
        self.assertIn("缺失参数", html)
        self.assertIn("workflow", html)
        self.assertIn("function sendKeyFields", html)
        self.assertIn("提交关键参数", html)

    def test_homepage_links_agent_train(self):
        html = (ROOT / "frontend" / "dist" / "index.html").read_text(encoding="utf-8")

        self.assertIn("/agent-train.html", html)
        self.assertIn("Agent Train", html)


if __name__ == "__main__":
    unittest.main()
