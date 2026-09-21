"""Static policy: fast, no Docker."""

import pytest

from tsagent.sandbox import DEFAULT_ALLOWED_MODULES, ViolationKind, check_code


def kinds(code):
    return [v.kind for v in check_code(code, DEFAULT_ALLOWED_MODULES).violations]


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nresult = 1",
        "import numpy as np\nfrom datetime import timedelta\nresult = np.mean([1, 2])",
        "result = df.resample('D').mean()['temp'].max()",
        "from pandas.api.types import is_numeric_dtype",
    ],
)
def test_legitimate_analysis_code_passes(code):
    assert check_code(code, DEFAULT_ALLOWED_MODULES).ok


@pytest.mark.parametrize(
    "code, expected",
    [
        ("import os", ViolationKind.STATIC_IMPORT),
        ("import os.path", ViolationKind.STATIC_IMPORT),
        ("from subprocess import run", ViolationKind.STATIC_IMPORT),
        ("from . import x", ViolationKind.STATIC_IMPORT),
        ("__import__('os')", ViolationKind.STATIC_FORBIDDEN_NAME),
        ("eval('1+1')", ViolationKind.STATIC_FORBIDDEN_NAME),
        ("open('/etc/passwd')", ViolationKind.STATIC_FORBIDDEN_NAME),
        ("getattr(pd, 'io')", ViolationKind.STATIC_FORBIDDEN_NAME),
        ("().__class__.__base__.__subclasses__()", ViolationKind.STATIC_FORBIDDEN_ATTR),
        ("f.__globals__", ViolationKind.STATIC_FORBIDDEN_ATTR),
    ],
)
def test_rejections(code, expected):
    assert expected in kinds(code)


def test_syntax_error_reported_not_raised():
    r = check_code("def f(:", DEFAULT_ALLOWED_MODULES)
    assert r.syntax_error and not r.ok


def test_known_bypass_is_documented_not_hidden():
    """The static check CANNOT see this. Kept as a test so the limitation is
    explicit; the container is what protects us here (see test_sandbox.py)."""
    code = "import pandas as pd\nresult = pd.io.common.os.getcwd()"
    assert check_code(code, DEFAULT_ALLOWED_MODULES).ok
