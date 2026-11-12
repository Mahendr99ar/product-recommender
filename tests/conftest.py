import copy

import pytest

from prodrec.config import Config, load_config
from prodrec.data.synthetic import generate


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Real config, but every path points into a temp folder and the data is small."""
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["synthetic"]["customers"] = 6000
    raw["model"]["num_boost_round"] = 150
    return Config(raw, root=tmp_path_factory.mktemp("prodrec"))


@pytest.fixture(scope="session")
def demo(cfg):
    generate(cfg)
    return cfg
