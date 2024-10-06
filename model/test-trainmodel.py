import pytest
from unittest.mock import patch, MagicMock
import transformers

# Assuming your main script is in a file called 'train_phi3.py'
from trainmodel import train_my_model, Phi3


@pytest.fixture
def mock_trainer():
    with patch("transformers.Trainer") as MockTrainer:
        mock_train = MagicMock()
        MockTrainer.return_value.train = mock_train
        yield MockTrainer


@pytest.fixture
def mock_mlflow():
    with patch("mlflow.log_metric") as mock_log_metric, patch(
        "mlflow.pyfunc.log_model"
    ) as mock_log_model:
        yield mock_log_metric, mock_log_model


def test_train_function_called(mock_trainer, mock_mlflow):
    # Mock the is_bfloat16_supported function
    with patch("train_phi3.is_bfloat16_supported", return_value=False):
        # Call the training function
        train_my_model()

    # Assert that the Trainer was instantiated
    mock_trainer.assert_called_once()

    # Assert that the train method was called on the Trainer instance
    mock_trainer.return_value.train.assert_called_once()

    # We're not asserting anything about the actual training process or results


if __name__ == "__main__":
    pytest.main()
