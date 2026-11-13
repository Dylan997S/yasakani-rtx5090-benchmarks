import ast
import builtins
import json
import sys


source, tests = json.load(sys.stdin)
tree = ast.parse(source)
allowed_modules = {"bisect", "collections", "functools", "heapq", "itertools", "math", "typing"}
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        if any(alias.name not in allowed_modules for alias in node.names):
            raise ValueError("import outside scoring allowlist")
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module not in allowed_modules or any(alias.name == "*" for alias in node.names):
            raise ValueError("import outside scoring allowlist")
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise ValueError("private attribute")
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        raise ValueError("private name")


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in allowed_modules:
        raise ImportError(name)
    return builtins.__import__(name, globals, locals, fromlist, level)


names = ("abs all any bool dict enumerate filter float int isinstance iter len list map max min next "
         "range reversed set sorted str sum tuple zip ValueError IndexError KeyError StopIteration object").split()
scope = {"__builtins__": {name: getattr(builtins, name) for name in names}}
scope["__builtins__"]["__import__"] = safe_import
exec(compile(tree, "<model answer>", "exec"), scope)
checks = [scope["solve"](*args) == expected for args, expected in tests]
print(json.dumps(checks))
