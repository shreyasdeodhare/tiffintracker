"""Registry edits are desired-state changes; activation always requires apply."""

import copy

from .storage import load, write_json
from .validator import Error, validate_registry


def read(path, roots=()):
    return validate_registry(load(path), roots)


def merge(original, patch):
    result = copy.deepcopy(original)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def edit(path, action, name=None, value=None, roots=()):
    registry = read(path, roots)
    apps = registry["applications"]
    index = next((i for i, app in enumerate(apps) if app["name"] == name), None)
    if action == "add":
        apps.append(value)
    elif index is None:
        raise Error("Application not found")
    elif action == "update":
        if not isinstance(value, dict) or "name" in value:
            raise Error("Update must be an object without name; rename is unsupported")
        apps[index] = merge(apps[index], value)
    elif action == "remove":
        apps.pop(index)
    elif action in ("enable", "disable"):
        apps[index]["enabled"] = action == "enable"
    else:
        raise Error("Unknown registry operation")
    validate_registry(registry, roots)
    write_json(path, registry)
    return registry
