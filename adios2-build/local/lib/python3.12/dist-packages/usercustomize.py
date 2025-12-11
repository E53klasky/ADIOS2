"""This script is used to setup the testing environment for the project."""

import sys

if sys.platform.startswith("win"):
    import os
    from pathlib import Path
    os.add_dll_directory(Path(__file__).parent)
    for build_type in "".split(";"):
        runtime_dir = Path("/home/jlx/Projects/CAESAR_ALL/ADIOS2/adios2-build/bin") / build_type
        if runtime_dir.is_dir():
            os.add_dll_directory(runtime_dir)
