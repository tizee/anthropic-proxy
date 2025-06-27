import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

# Define hook specifications
request_hook_spec = "request_hook"
response_hook_spec = "response_hook"


class HookManager:
    def __init__(self):
        self.request_hooks: list[Callable[[dict[str, Any]], dict[str, Any]]] = []
        self.response_hooks: list[Callable[[dict[str, Any]], dict[str, Any]]] = []

    def load_plugins(self, package):
        """Loads plugins from the given package."""
        if not hasattr(package, "__path__"):
            print(f"Package {package.__name__} does not have a __path__.")
            return

        for _, name, _ in pkgutil.iter_modules(package.__path__):
            try:
                module = importlib.import_module(f"{package.__name__}.{name}")
                self.register_hooks(module)
            except Exception as e:
                print(f"Failed to load plugin {name}: {e}")

    def register_hooks(self, module):
        """Registers hooks from a loaded module."""
        if hasattr(module, request_hook_spec):
            self.request_hooks.append(getattr(module, request_hook_spec))
        if hasattr(module, response_hook_spec):
            self.response_hooks.append(getattr(module, response_hook_spec))

    def trigger_request_hooks(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Triggers all registered request hooks on a payload."""
        for hook in self.request_hooks:
            payload = hook(payload)
        return payload

    def trigger_response_hooks(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Triggers all registered response hooks on a payload."""
        for hook in self.response_hooks:
            payload = hook(payload)
        return payload


hook_manager = HookManager()


def load_all_plugins():
    """Discover and load all plugins from the 'plugins' directory."""
    import plugins

    hook_manager.load_plugins(plugins)
