import pytest
import sys
sys.path.append("YoloV3_PyTorch")
from train import *

class TestRun:
    @pytest.fixture(scope="session")
    def model(pytestconfig):
        return pytestconfig.getoption("model")
    @pytest.fixture(scope="session")
    def data(pytestconfig):
        return pytestconfig.getoption("data")

    @pytest.fixture(scope="session")
    def epochs(pytestconfig):
        return pytestconfig.getoption("epochs")

    @pytest.fixture(scope="session")
    def pretrained_weights(pytestconfig):
        return pytestconfig.getoption("pretrained_weights")

    @pytest.fixture(scope="session")
    def evaluation_interval(pytestconfig):
        return pytestconfig.getoption("evaluation_interval")
    @pytest.fixture(scope="session")
    def gpu(pytestconfig):
        return pytestconfig.getoption("gpu")
    def test_run(self):
       assert run() == 'Maximum number of batches reached'