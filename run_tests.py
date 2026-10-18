import torch
import torch.fx
import sys
import traceback

def run_all_tests():
    print("======================================================================")
    print("RUNNING SMART FOREST AUTOMATED TEST SUITE")
    print("======================================================================")
    
    test_files = [
        ("Geofence Engine", "backend.tests.test_geofence"),
        ("Risk Engine", "backend.tests.test_risk_engine"),
        ("Authentication & JWT", "backend.tests.test_auth"),
        ("SOS Video & SHA-256", "backend.tests.test_sos_video"),
        ("Vision Pipeline & ML", "backend.tests.test_vision_pipeline")
    ]

    total_passed = 0
    total_failed = 0

    for name, module_name in test_files:
        print(f"\n[TEST SUITE] {name} ({module_name}):")
        try:
            mod = __import__(module_name, fromlist=["*"])
            test_funcs = [getattr(mod, f) for f in dir(mod) if f.startswith("test_") and callable(getattr(mod, f))]
            
            for tf in test_funcs:
                try:
                    tf()
                    print(f"  [PASS] {tf.__name__}")
                    total_passed += 1
                except Exception as e:
                    print(f"  [FAIL] {tf.__name__}: {e}")
                    traceback.print_exc()
                    total_failed += 1
        except Exception as e:
            print(f"  [ERROR LOADING SUITE] {module_name}: {e}")
            traceback.print_exc()
            total_failed += 1

    print("\n======================================================================")
    print(f"TEST RESULTS: {total_passed} PASSED, {total_failed} FAILED")
    print("======================================================================")
    
    if total_failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
