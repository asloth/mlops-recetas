from datasets import load_dataset
import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock the entire unsloth module
mock_unsloth = MagicMock()
mock_FastLanguageModel = MagicMock()

# Create mock objects for model and tokenizer
mock_model = MagicMock()
mock_tokenizer = MagicMock()

# Configure the from_pretrained method to return the mock model and tokenizer
mock_FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)

mock_unsloth.FastLanguageModel = mock_FastLanguageModel
sys.modules["unsloth"] = mock_unsloth
sys.modules["unsloth.FastLanguageModel"] = mock_FastLanguageModel

# Now patch the import in your script
with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
    # Import your script here
    from trainmodel import train_my_model


@pytest.fixture
def mock_trainer():
    with patch("trainmodel.SFTTrainer") as MockTrainer:
        mock_train = MagicMock()
        mock_train.return_value.metrics = {"train_runtime": 600}  # 10 minutes
        MockTrainer.return_value.train = mock_train
        yield MockTrainer


@pytest.fixture
def mock_mlflow():
    with patch("mlflow.start_run") as mock_start_run, patch(
        "mlflow.log_param"
    ) as mock_log_param, patch("mlflow.log_metric") as mock_log_metric, patch(
        "mlflow.pyfunc.log_model"
    ) as mock_log_model:
        yield mock_start_run, mock_log_param, mock_log_metric, mock_log_model


mock_FastLanguageModel = MagicMock()
mock_model = MagicMock()
mock_tokenizer = MagicMock()
mock_FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)


@patch("trainmodel.SFTTrainer")
# @patch("trainmodel.trainer.train")
@patch("trainmodel.TrainingArguments")
@patch("trainmodel.FastLanguageModel", return_value=mock_FastLanguageModel)
# @patch("trainmodel.model")
def test_train_my_model(
    # mock_model,
    mock_fast_language_model,
    mock_training_args,
    # mock_trainer_train,
    mock_sfttrainer,
    mock_trainer,
    mock_mlflow,
):
    mock_start_run, mock_log_param, mock_log_metric, mock_log_model = mock_mlflow

    train_my_model()
    mock_model.assert_called_once()
    mock_training_args.assert_called_once()
    mock_trainer.assert_called_once()
    mock_log_model.assert_called_once()


if __name__ == "__main__":
    pytest.main()
