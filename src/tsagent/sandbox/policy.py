"""Static policy check, run on the host before any container starts.

What it is: a cheap filter that catches the common ways generated code goes
outside the intended contract, returns a precise message the model can act on,
and records a countable violation for the eval harness.

What it is NOT: a security boundary. Python cannot be reliably sandboxed at the
language level (e.g. reaching `os` through `pandas.io.common.os` needs no import
and no dunder). The container is the boundary; tests/test_sandbox.py disables
this layer and checks the container alone still holds.
"""

import ast
from dataclasses import dataclass, field

from .models import Violation, ViolationKind

# Reflection / dynamic execution. Legitimate pandas analysis code essentially never needs these.
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "vars",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "open",
    }
)

FORBIDDEN_ATTRS = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__base__",
        "__bases__",
        "__mro__",
        "__loader__",
        "__spec__",
        "__import__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
    }
)


@dataclass
class PolicyReport:
    syntax_error: str | None = None
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.syntax_error is None and not self.violations

    def message_for_model(self) -> str:
        if self.syntax_error:
            return f"SyntaxError: {self.syntax_error}"
        lines = ["Code rejected by sandbox policy before execution:"]
        lines += [f"  line {v.line}: {v.kind.value}: {v.detail}" for v in self.violations]
        return "\n".join(lines)


def check_code(code: str, allowed_modules: frozenset[str]) -> PolicyReport:
    report = PolicyReport()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        report.syntax_error = f"{e.msg} (line {e.lineno})"
        return report

    def add(kind: ViolationKind, detail: str, node: ast.AST) -> None:
        report.violations.append(Violation(kind=kind, detail=detail, line=getattr(node, "lineno", None)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] not in allowed_modules:
                    add(ViolationKind.STATIC_IMPORT, f"import {alias.name}", node)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0 or module.partition(".")[0] not in allowed_modules:
                add(ViolationKind.STATIC_IMPORT, f"from {'.' * node.level}{module} import ...", node)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            add(ViolationKind.STATIC_FORBIDDEN_NAME, node.id, node)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            add(ViolationKind.STATIC_FORBIDDEN_ATTR, node.attr, node)

    return report
