import pytest
from unittest.mock import patch, MagicMock
import mlflow
import transformers

# Assuming your main script is in a file called 'train_phi3.py'
from trainmodel import train_my_model, Phi3


@pytest.fixture
def mock_trainer():
    with patch("transformers.Trainer") as MockTrainer:
        mock_train = MagicMock()
        mock_train.return_value.metrics = {"train_runtime": 600}  # 10 minutes
        MockTrainer.return_value.train = mock_train
        yield MockTrainer


@pytest.fixture
def mock_mlflow():
    with patch("mlflow.log_metric") as mock_log_metric, patch(
        "mlflow.pyfunc.log_model"
    ) as mock_log_model:
        yield mock_log_metric, mock_log_model


def test_train_phi3(mock_trainer, mock_mlflow):
    mock_log_metric, mock_log_model = mock_mlflow

    # Mock the is_bfloat16_supported function
    with patch("train_phi3.is_bfloat16_supported", return_value=False):
        # Call the training function
        train_my_model()

    # Assert that the trainer was called with the correct arguments
    mock_trainer.assert_called_once()
    trainer_args = mock_trainer.call_args[1]["args"]
    assert trainer_args.learning_rate == 2e-4
    assert trainer_args.fp16 == True
    assert trainer_args.bf16 == False
    assert trainer_args.logging_steps == 1
    assert trainer_args.optim == "adamw_8bit"
    assert trainer_args.lr_scheduler_type == "linear"
    assert trainer_args.seed == 3407
    assert trainer_args.output_dir == "outputs"

    # Assert that the training was performed
    mock_trainer.return_value.train.assert_called_once()

    # Assert that MLflow metrics were logged correctly
    mock_log_metric.assert_called_once_with("minutesxtraining", 10.0)

    # Assert that the model was saved and logged with MLflow
    mock_log_model.assert_called_once()
    log_model_args = mock_log_model.call_args[1]
    assert log_model_args["artifact_path"] == "phi3-instruct"
    assert isinstance(log_model_args["python_model"], Phi3)
    assert log_model_args["registered_model_name"] == "PhiModel"


if __name__ == "__main__":
    pytest.main()
