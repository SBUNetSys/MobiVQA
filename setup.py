#!/usr/bin/env python
import platform
from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def _load_requirements(req_file, comment_char="#"):
    with open(req_file, "r") as f:
        reqs = {
            line.strip()
            for line in f.readlines()
            if line.strip() and comment_char not in line.strip()
        }
    return reqs


PATH_ROOT = Path(__file__).parent

system_type = platform.release()
req_folder = PATH_ROOT / "requirements"
deps_file = req_folder / "deps.txt"
deps_reqs = _load_requirements(deps_file)

sys_suffix = "tx2" if system_type.endswith("tegra") else "pc"
sys_req_file = req_folder / f"req-{sys_suffix}.txt"
extra_reqs = _load_requirements(sys_req_file)
all_reqs = deps_reqs.union(extra_reqs)

setup(
    name="vqa",
    version="0.1.0",
    python_requires=">=3.6",
    install_requires=list(all_reqs),
    packages=find_packages(exclude=("data", "notes")),
)
