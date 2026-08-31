import sys
from pathlib import Path

in_path = sys.argv[1]

# Replace with your actual directory path
target_path = Path(in_path)

ac_only = """1/2: Building AlternativeCore (AlternativeCore.idr)
2/2: Building DerivedGen (DerivedGen.idr)
"""

rdg_only = """1/2: Building RunDerivedGen (RunDerivedGen.idr)
2/2: Building DerivedGen (DerivedGen.idr)
"""


ac_rdg = """1/3: Building AlternativeCore (AlternativeCore.idr)
2/3: Building RunDerivedGen (RunDerivedGen.idr)
3/3: Building DerivedGen (DerivedGen.idr)
"""

dg_only = """1/1: Building DerivedGen (DerivedGen.idr)
"""

infra = """1/2: Building Infra (Infra.idr)
2/2: Building CanonicSigCheck (CanonicSigCheck.idr)
"""

csc_only = """1/1: Building CanonicSigCheck (CanonicSigCheck.idr)
"""


def patch_expected(content: str) -> str:
    content = content.replace(ac_only, dg_only)
    content = content.replace(rdg_only, dg_only)
    content = content.replace(ac_rdg, dg_only)
    content = content.replace(infra, csc_only)
    return content


for item in target_path.rglob("*"):
    if item.is_dir() and item.name != "_shared" and item.name != "_common":
        print(f"Directory found: {item}")
        (item / "AlternativeCore.idr").unlink(missing_ok=True)
        (item / "RunDerivedGen.idr").unlink(missing_ok=True)
        (item / "Infra.idr").unlink(missing_ok=True)
        expected = item / "expected"
        if expected.exists():
            content = ""
            with expected.open("r") as f:
                content = f.read()
            content = patch_expected(content)
            with expected.open("w") as f:
                f.write(content)
