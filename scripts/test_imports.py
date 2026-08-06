#!/usr/bin/env python3
"""
Verify all required imports and environment setup.

Usage:
    python scripts/test_imports.py
"""

import sys
import traceback

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def test_import(module_name: str, import_statement: str = None) -> bool:
    """Test if a module can be imported. Returns True on success."""
    if import_statement is None:
        import_statement = f"import {module_name}"
    try:
        exec(import_statement)
        print(f"  {PASS} {module_name}")
        return True
    except ImportError as e:
        print(f"  {FAIL} {module_name} — {e}")
        return False
    except Exception as e:
        print(f"  {WARN} {module_name} — unexpected: {e}")
        return False


def main():
    print("=" * 60)
    print("  Catan RL + LLM — Environment Verification")
    print("=" * 60)

    results = {}

    # ---- Python Version ----
    print("\n[1] Python Version")
    py_ver = sys.version_info
    print(f"  Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    ok = py_ver >= (3, 10)
    print(f"  {'[PASS]' if ok else '[FAIL]'} Requires >= 3.10")
    results["python"] = ok

    # ---- Core ML Libraries ----
    print("\n[2] Core ML Libraries")
    results["torch"] = test_import("torch")
    results["transformers"] = test_import("transformers")
    results["trl"] = test_import("trl")
    results["peft"] = test_import("peft")
    results["accelerate"] = test_import("accelerate")
    results["bitsandbytes"] = test_import("bitsandbytes")
    results["datasets"] = test_import("datasets")

    # ---- Game Environment ----
    print("\n[3] Game Environment")
    results["catanatron"] = test_import("catanatron")
    results["catanatron_gym"] = test_import("catanatron_gym")
    results["gymnasium"] = test_import("gymnasium")

    # ---- Utilities ----
    print("\n[4] Utilities")
    results["yaml"] = test_import("yaml")
    results["tqdm"] = test_import("tqdm")
    results["wandb"] = test_import("wandb")
    results["matplotlib"] = test_import("matplotlib")
    results["seaborn"] = test_import("seaborn")

    # ---- GPU Check ----
    print("\n[5] GPU Check")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  {PASS} CUDA available")
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")
            results["gpu"] = True
        else:
            print(f"  {WARN} CUDA not available — training will be very slow on CPU")
            results["gpu"] = False
    except Exception as e:
        print(f"  {FAIL} GPU check failed: {e}")
        results["gpu"] = False

    # ---- Catanatron Env Test ----
    print("\n[6] Catanatron Environment Quick Test")
    try:
        from catanatron_gym.envs.catanatron_env import CatanatronEnv
        env = CatanatronEnv(config={
            "map_type": "MINI",
            "vps_to_win": 6,
        })
        obs = env.reset()
        valid_actions = env.get_valid_actions()
        print(f"  {PASS} Environment created: MINI map, 6 VP")
        print(f"  Action space: {env.action_space}")
        print(f"  Valid actions at start: {len(valid_actions)}")
        # Test step with a valid action (returns 5 values: gymnasium format)
        step_result = env.step(valid_actions[0])
        obs, reward, terminated, truncated, info = step_result
        print(f"  Step OK — reward: {reward}, done: {terminated or truncated}")
        env.close()
        results["catan_env"] = True
    except Exception as e:
        print(f"  {FAIL} Environment test failed: {e}")
        traceback.print_exc()
        results["catan_env"] = False

    # ---- Summary ----
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  Summary: {passed}/{total} checks passed")

    if passed == total:
        print("  All checks passed! Environment is ready.")
        return 0
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed: {', '.join(failed)}")
        print("  Run setup: bash scripts/setup_env.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
