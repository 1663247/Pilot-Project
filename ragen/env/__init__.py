"""Project environment registry: load only the selected environment's dependencies."""
from collections.abc import Mapping
from importlib import import_module


class _EnvironmentRegistry(Mapping):
    def __init__(self, entries):
        self._entries = entries
        self._loaded = {}

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, name):
        module, attribute = self._entries[name]
        if name not in self._loaded:
            try:
                self._loaded[name] = getattr(import_module(module), attribute)
            except ImportError as exc:
                raise ImportError(
                    f"Cannot load {name!r}: install its environment dependencies first. "
                    f"For WebShop, initialize external/webshop-minimal and follow its setup guide."
                ) from exc
        return self._loaded[name]

    def __contains__(self, name):
        return name in self._entries


REGISTERED_ENVS = _EnvironmentRegistry({
    'sokoban': ('ragen.env.sokoban.env', 'SokobanEnv'),
    'frozen_lake': ('ragen.env.frozen_lake.env', 'FrozenLakeEnv'),
    'webshop': ('ragen.env.webshop.env', 'WebShopEnv'),
})
REGISTERED_ENV_CONFIGS = _EnvironmentRegistry({
    'sokoban': ('ragen.env.sokoban.config', 'SokobanEnvConfig'),
    'frozen_lake': ('ragen.env.frozen_lake.config', 'FrozenLakeEnvConfig'),
    'webshop': ('ragen.env.webshop.config', 'WebShopEnvConfig'),
})
