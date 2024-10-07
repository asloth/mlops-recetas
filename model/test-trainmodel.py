import pytest
from unittest.mock import patch, MagicMock
from transformers import TrainingArguments
from datasets import Dataset
import sys
from trl import SFTTrainer
import mlflow
from unittest.mock import create_autospec


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
    from trainmodel import (
        train_my_model,
        formatting_prompts_func,
        Phi3,
        is_bfloat16_supported,
    )


@pytest.fixture
def mock_dataset():
    return MagicMock(spec=Dataset)


class PicklableMock:
    def train(self):
        return {"train_runtime": 600}  # 10 minutes


# @pytest.fixture
# def mock_trainer():
#    with patch("trl.SFTTrainer", return_value=PicklableMock()) as MockTrainer:
#        yield MockTrainer


@pytest.fixture
def start_run():
    mlflow.start_run()
    yield
    mlflow.end_run()


# @pytest.fixture
# def mock_mlflow():
#    mock_run = MagicMock()
#    mock_mlflow = MagicMock()
#    with patch("mlflow.start_run", return_value=mock_run) as mock_start_run, patch(
#        "mlflow.log_param"
#    ) as mock_log_param, patch("mlflow.log_metric") as mock_log_metric, patch(
#        "mlflow.pyfunc.log_model"
#    ) as mock_log_model, patch(
#        "mlflow.MlflowClient", return_value=mock_mlflow
#    ) as mock_mlflowclient, patch(
#        "mlflow.set_experiment"
#    ) as mock_set_experiment, patch(
#        "mlflow.set_tracking_uri"
#    ) as mock_set_tracking_uri:
#        yield mock_start_run, mock_log_param, mock_log_metric, mock_log_model, mock_mlflowclient, mock_set_experiment, mock_set_tracking_uri
#
# Mock the necessary modules and functions
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, patch(
        "transformers.AutoModelForCausalLM.from_pretrained"
    ) as mock_model, patch("datasets.load_dataset") as mock_load_dataset, patch(
        "mlflow.start_run", return_value=MagicMock()
    ), patch(
        "mlflow.MlflowClient", return_value=MagicMock()
    ), patch(
        "mlflow.set_tracking_uri"
    ), patch(
        "mlflow.set_experiment"
    ), patch(
        "mlflow.log_param"
    ), patch(
        "mlflow.log_metric"
    ), patch(
        "mlflow.pyfunc.log_model"
    ):

        # Set up mock returns
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_load_dataset.return_value = MagicMock()

        yield


def test_train_my_model():
    # Mock the global variables that are used in the function
    global model, tokenizer, max_seq_length
    model = MagicMock()
    tokenizer = MagicMock()
    max_seq_length = 512

    # Mock the formatting_prompts_func
    with patch("__main__.formatting_prompts_func", return_value=MagicMock()):
        # Mock the is_bfloat16_supported function
        with patch("__main__.is_bfloat16_supported", return_value=False):
            # Create a mock trainer with a mock train method
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = MagicMock(metrics={"train_runtime": 600})

            # Patch the SFTTrainer to return our mock trainer
            with patch("trl.SFTTrainer", return_value=mock_trainer):
                # Call the function we want to test
                train_my_model()

    # Assert that MLflow functions were called
    mlflow.set_tracking_uri.assert_called_once_with("http://mlflow-server:5000")
    mlflow.set_experiment.assert_called_once_with("recetas-model")

    # Check that log_param was called for each parameter
    mlflow.log_param.assert_any_call("warmup_steps", 5)
    mlflow.log_param.assert_any_call("per_device_train_batch_size", 2)
    mlflow.log_param.assert_any_call("max_steps", 10)
    mlflow.log_param.assert_any_call("weight_decay", 0.01)

    # Check that log_metric was called with the correct value
    mlflow.log_metric.assert_called_once_with("minutesxtraining", 10.0)

    # Check that the model was saved
    model.save_pretrained_merged.assert_called_once()

    # Check that the model was logged to MLflow
    mlflow.pyfunc.log_model.assert_called_once()


# You would typically put this in a separate file, but for demonstration:
if __name__ == "__main__":
    pytest.main([__file__])
