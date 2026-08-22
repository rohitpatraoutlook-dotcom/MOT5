"""
MOT5 Test Runner - Organized Test Suite
"""
import os
import sys
import subprocess
from datetime import datetime

def run_test(test_file, description):
    """Run a single test file"""
    print(f"\n{'='*60}")
    print(f"🔬 {description}")
    print(f"📁 {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run(
            ['python3', test_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        print(result.stdout)
        if result.stderr:
            print("WARNINGS:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (>120s)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    print("="*60)
    print("MOT5 TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    tests = [
        ("tests/regression/test_piecewise.py", "Piecewise Function (abs)"),
        ("tests/regression/test_nested_v4.py", "Nested Function (sin(cos(x)))"),
    ]
    
    results = {}
    
    for test_file, description in tests:
        if not os.path.exists(test_file):
            print(f"\n⚠️ Test file not found: {test_file}")
            results[description] = "SKIPPED"
            continue
        
        success = run_test(test_file, description)
        results[description] = "✅ PASSED" if success else "❌ FAILED"
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, status in results.items():
        print(f"{status}  {name}")
    
    passed = sum(1 for s in results.values() if "PASSED" in s)
    total = len(results)
    print(f"\nScore: {passed}/{total} ({passed/total*100:.1f}%)")

if __name__ == '__main__':
    main()
