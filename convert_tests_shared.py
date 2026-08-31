"""convert_tests_shared.py

This script automatically removes symlinks to shared test code that is now
managed by pack. It's left in the git history for posterity.
"""

import sys
from pathlib import Path

in_path = sys.argv[1]

# Replace with your actual directory path
target_path = Path(in_path)

AC_ONLY = """1/2: Building AlternativeCore (AlternativeCore.idr)
2/2: Building DerivedGen (DerivedGen.idr)
"""

RDG_ONLY = """1/2: Building RunDerivedGen (RunDerivedGen.idr)
2/2: Building DerivedGen (DerivedGen.idr)
"""


AC_RDG = """1/3: Building AlternativeCore (AlternativeCore.idr)
2/3: Building RunDerivedGen (RunDerivedGen.idr)
3/3: Building DerivedGen (DerivedGen.idr)
"""

DG_ONLY = """1/1: Building DerivedGen (DerivedGen.idr)
"""

INFRA_CSC = """1/2: Building Infra (Infra.idr)
2/2: Building CanonicSigCheck (CanonicSigCheck.idr)
"""

CSC_ONLY = """1/1: Building CanonicSigCheck (CanonicSigCheck.idr)
"""

SHARED_TEST = """1/2: Building Shared (Shared.idr)
2/2: Building Test (Test.idr)"""

TEST_ONLY = """1/1: Building Test (Test.idr)"""


def patch_expected(s: str) -> str:
    """Remove references to deleted symlinks from golden test expected file"""
    return (
        s.replace(AC_ONLY, DG_ONLY)
        .replace(RDG_ONLY, DG_ONLY)
        .replace(AC_RDG, DG_ONLY)
        .replace(INFRA_CSC, CSC_ONLY)
        .replace(SHARED_TEST, TEST_ONLY)
    )


for item in target_path.rglob("*"):
    if item.is_dir() and item.name != "_shared" and item.name != "_common":
        print(f"Directory found: {item}")
        (item / "AlternativeCore.idr").unlink(missing_ok=True)
        (item / "RunDerivedGen.idr").unlink(missing_ok=True)
        (item / "Infra.idr").unlink(missing_ok=True)
        (item / "DistrCheckCommon.idr").unlink(missing_ok=True)
        (item / "Shared.idr").unlink(missing_ok=True)
        expected = item / "expected"
        if expected.exists():
            content = ""
            with expected.open("r") as f:
                content = f.read()
            content = patch_expected(content)
            with expected.open("w") as f:
                f.write(content)
