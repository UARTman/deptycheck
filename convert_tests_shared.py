"""convert_tests_shared.py

This script automatically removes symlinks to shared test code that is now
managed by pack. It's left in the git history for posterity.
"""

import sys
from collections.abc import Callable
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

CA_INFRA = """1/2: Building ConsApps (ConsApps.idr)
2/2: Building Infra (Infra.idr)"""

CA_ONLY = """1/1: Building ConsApps (ConsApps.idr)"""


def patch_expected(s: str) -> str:
    """Remove references to deleted symlinks from golden test expected file"""
    return (
        s.replace(AC_ONLY, DG_ONLY)
        .replace(RDG_ONLY, DG_ONLY)
        .replace(AC_RDG, DG_ONLY)
        .replace(INFRA_CSC, CSC_ONLY)
        .replace(SHARED_TEST, TEST_ONLY)
        .replace(CA_INFRA, CA_ONLY)
    )


CONS_APPS_BEFORE = """module ConsApps

import Language.Reflection.Compat

%default total"""

CONS_APPS_AFTER = """module ConsApps

import Infra

import Language.Reflection.Compat

%default total

%language ElabReflection

%hide Data.List.Quantifiers.Right
%hide Data.List.Quantifiers.Left"""

CONS_APPS_DERIVE = """
%runElab consApps >>= traverse_ (uncurry printDeepConsApp)
"""


def patch_cons_apps(s: str) -> str:
    """Patch ConsApps.idr"""
    s_new = s.replace(CONS_APPS_BEFORE, CONS_APPS_AFTER)
    if CONS_APPS_DERIVE not in s_new:
        s_new = s_new + CONS_APPS_DERIVE
    return s_new


def patch_file(fl: Path, patch_fn: Callable[[str], str]):
    """Patch a file with a patcher function"""
    if fl.exists():
        content = ""
        with fl.open("r") as f:
            content = f.read()
        content = patch_fn(content)
        with fl.open("w") as f:
            f.write(content)


for item in target_path.rglob("*"):
    if (
        item.is_dir()
        and item.name != "_shared"
        and item.name != "_common"
        and item.name != "_common-deep-cons-app"
    ):
        print(f"Directory found: {item}")
        (item / "AlternativeCore.idr").unlink(missing_ok=True)
        (item / "RunDerivedGen.idr").unlink(missing_ok=True)
        (item / "Infra.idr").unlink(missing_ok=True)
        (item / "DistrCheckCommon.idr").unlink(missing_ok=True)
        (item / "Shared.idr").unlink(missing_ok=True)
        patch_file(item / "expected", patch_expected)
        patch_file(item / "ConsApps.idr", patch_cons_apps)
