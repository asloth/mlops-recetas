import pytest
from unittest.mock import patch, MagicMock
from transformers import TrainingArguments
from datasets import Dataset
import sys
import mlflow

mock_mlflow = MagicMock()
# Mock FastLanguageModel
mock_model = MagicMock()
mock_tokenizer = MagicMock()
mock_FastLanguageModel = MagicMock()
mock_FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)


mock_unsloth = MagicMock()
mock_unsloth.FastLanguageModel = mock_FastLanguageModel
# Patch necessary modules and imports
# @pytest.fixture(autouse=True)
# def mock_imports():
with patch.dict(
    "sys.modules",
    {
        "mlflow": MagicMock(),
        "unsloth": mock_unsloth,
        "unsloth.FastLanguageModel": mock_FastLanguageModel,
        "unsloth.is_bfloat16_supported": False,
    },
):
    from trainmodel import (
        train_my_model,
        formatting_prompts_func,
        Phi3,
    )


@pytest.fixture
def mock_dataset():
    return MagicMock(spec=Dataset)


@pytest.fixture
def mock_trainer():
    with patch("trl.SFTTrainer") as MockTrainer:
        mock_train = MagicMock()
        mock_train.return_value.metrics = {"train_runtime": 600}  # 10 minutes
        MockTrainer.return_value.train = mock_train
        yield MockTrainer


def test_train_my_model(mock_dataset, mock_trainer):
    # Mock the load_dataset function
    with patch("datasets.load_dataset", return_value=mock_dataset):
        # Mock the is_bfloat16_supported function
        # with patch("unsloth.is_bfloat16_supported", return_value=False):
        # Call the training function
        train_my_model()

    # Assert that FastLanguageModel.from_pretrained was called
    mock_FastLanguageModel.from_pretrained.assert_called_once()

    # Assert that mlflow.start_run was called
    mock_mlflow.start_run.assert_called_once()

    # Assert that the dataset.map method was called with formatting_prompts_func
    mock_dataset.map.assert_called_once_with(formatting_prompts_func, batched=True)

    # Assert that the training was performed
    mock_trainer.return_value.train.assert_called_once()

    # Assert that the model was saved
    mock_model.save_pretrained_merged.assert_called_once_with(
        "model",
        mock_tokenizer,
        save_method="merged_16bit",
    )

    # Assert that the model was logged with MLflow
    mock_mlflow.pyfunc.log_model.assert_called_once()
    log_model_args = mock_mlflow.pyfunc.log_model.call_args[1]
    assert log_model_args["artifact_path"] == "phi3-instruct"
    assert isinstance(log_model_args["python_model"], Phi3)
    assert log_model_args["artifacts"] == {"snapshot": "./model"}
    assert log_model_args["registered_model_name"] == "PhiModel"


if __name__ == "__main__":
    pytest.main()
