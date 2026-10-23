import pytest
import torch
import numpy as np
import random

from deepweeds.utils import set_seed, get_device, get_logger


class TestSetSeed:
    def test_sets_python_random_seed(self):
        set_seed(42)
        val1 = random.random()
        
        set_seed(42)
        val2 = random.random()
        
        assert val1 == val2

    def test_sets_numpy_seed(self):
        set_seed(123)
        val1 = np.random.rand()
        
        set_seed(123)
        val2 = np.random.rand()
        
        assert val1 == val2

    def test_sets_torch_seed(self):
        set_seed(456)
        val1 = torch.rand(1).item()
        
        set_seed(456)
        val2 = torch.rand(1).item()
        
        assert val1 == val2

    def test_deterministic_flag_sets_cudnn(self):
        # Just verify it runs without error
        set_seed(42, deterministic=True)
        set_seed(42, deterministic=False)


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cuda_if_available(self):
        device = get_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"

    def test_logger_has_handler(self):
        logger = get_logger("test_logger2")
        assert len(logger.handlers) > 0

    def test_same_name_returns_same_logger(self):
        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")
        assert logger1 is logger2

    def test_log_level_is_info(self):
        logger = get_logger("test_level")
        assert logger.level == 20  # logging.INFO