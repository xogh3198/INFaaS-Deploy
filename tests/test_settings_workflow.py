"""Host settings/Compose and script contract checks; no CUDA or Docker required."""
import importlib.util
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("infaas_settings_compose", ROOT / "deploy/docker/gen_compose.py")
compose = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compose
spec.loader.exec_module(compose)


class SettingsWorkflowTests(unittest.TestCase):
    def test_query_defaults_are_host_only_not_serving_env(self):
        cfg = compose.read_env(ROOT / "deploy/docker/infaas-docker.env")
        cfg.update(GPUS="2:v100", HW_COST="v100:1", MODEL_NAME="resnet-18",
                   INPUT_IMAGE="inputs/test.jpg", SLO_MS="300")
        smi = [{"index": 2, "uuid": "GPU-fixture", "name": "Tesla V100-DGXS-32GB", "mem_mib": 32768}]
        document, env, _, _ = compose.build(cfg, smi)
        worker = document["services"]["infaas-worker-2"]
        self.assertEqual(worker["environment"]["GPU_TYPE"], "v100")
        for key in ("MODEL_NAME", "INPUT_IMAGE", "SLO_MS"):
            self.assertNotIn(key, env)

    def test_local_selection_query_defaults_and_compatible_health(self):
        script = (ROOT / "deploy/docker/infaas-docker.sh").read_text(encoding="utf-8")
        self.assertIn("infaas-docker.local.env", script)
        self.assertIn('get MODEL_NAME resnet-50', script)
        self.assertIn('get INPUT_IMAGE inputs/test-dog.jpg', script)
        self.assertIn('get SLO_MS 200', script)
        self.assertIn('docker inspect --format', script)
        self.assertNotIn("ps --format '{{.Name}}", script)
        self.assertIn('workers did not become healthy', script)
        self.assertNotIn("export DOCKER_CONTEXT", script)

    def test_relative_mounts_resolve_from_repository_not_generated_directory(self):
        cfg = {"GPUS": "2:v100", "HW_COST": "v100:1", "MODEL_DIR": "data/infaas/models",
               "PROFILE_DIR": "data/infaas/profiles", "LOG_DIR": "data/infaas/logs"}
        smi = [{"index": 2, "uuid": "GPU-fixture", "name": "Tesla V100-DGXS-32GB", "mem_mib": 32768}]
        document, _, _, _ = compose.build(cfg, smi)
        volume = document["services"]["infaas-worker-2"]["volumes"][0]
        self.assertEqual(volume, str((ROOT / "data/infaas/models").resolve()) + ":/models")

    def test_setup_links_and_local_files_excluded(self):
        text = (ROOT / "setting_readme.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("```") % 2, 0)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target:
                self.assertTrue((ROOT / target.split("#")[0]).is_file(), target)
        self.assertIn("batch 1", text)
        self.assertIn("infaas-docker.local.env", (ROOT / ".gitignore").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
