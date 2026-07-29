import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")
sys.path.insert(0, PYTHON_DIR)


@pytest.fixture(scope="session")
def implementations():
    import editdistance as mojo_editdistance

    saved_path = sys.path[:]
    saved_module = sys.modules.pop("editdistance")
    sys.path = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.curdir) != os.path.abspath(PYTHON_DIR)
    ]
    try:
        import editdistance as upstream_editdistance
    finally:
        sys.path = saved_path
        sys.modules["editdistance"] = saved_module
    return mojo_editdistance, upstream_editdistance
