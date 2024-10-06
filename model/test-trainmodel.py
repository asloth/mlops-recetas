import pytest
from unittest.mock import patch, MagicMock
import mlflow
from transformers import TrainingArguments
from datasets import Dataset
import transformers

# Assuming your main script is in a file called 'train_script.py'
from train_script import (
    train_my_model,
    formatting_prompts_func,
    Phi3,
    is_bfloat16_supported,
)


@pytest.fixture
def mock_dataset():
    return MagicMock(spec=Dataset)


@pytest.fixture
def mock_trainer():
    with patch("train_script.SFTTrainer") as MockTrainer:
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


@pytest.fixture
def mock_model():
    with patch("train_script.model") as mock_model:
        yield mock_model


def test_train_my_model(mock_dataset, mock_trainer, mock_mlflow, mock_model):
    mock_start_run, mock_log_param, mock_log_metric, mock_log_model = mock_mlflow

    # Mock the load_dataset function
    with patch("train_script.load_dataset", return_value=mock_dataset):
        # Mock the is_bfloat16_supported function
        with patch("train_script.is_bfloat16_supported", return_value=False):
            # Call the training function
            train_my_model()

    # Assert that mlflow.start_run was called
    mock_start_run.assert_called_once()

    # Assert that mlflow.log_param was called with the correct parameters
    expected_params = {
        "warmup_steps": 5,
        "per_device_train_batch_size": 2,
        "max_steps": 10,
        "weight_decay": 0.01,
    }
    for param, value in expected_params.items():
        mock_log_param.assert_any_call(param, value)

    # Assert that the dataset.map method was called with formatting_prompts_func
    mock_dataset.map.assert_called_once_with(formatting_prompts_func, batched=True)

    # Assert that the Trainer was instantiated with the correct arguments
    mock_trainer.assert_called_once()
    trainer_args = mock_trainer.call_args[1]
    assert trainer_args["model"] == mock_model
    assert trainer_args["train_dataset"] == mock_dataset
    assert trainer_args["dataset_text_field"] == "text"
    assert (
        trainer_args["max_seq_length"] == max_seq_length
    )  # Assuming max_seq_length is defined in your script
    assert trainer_args["dataset_num_proc"] == 2
    assert trainer_args["packing"] == False

    # Check TrainingArguments
    assert isinstance(trainer_args["args"], TrainingArguments)
    assert trainer_args["args"].per_device_train_batch_size == 2
    assert trainer_args["args"].gradient_accumulation_steps == 4
    assert trainer_args["args"].warmup_steps == 5
    assert trainer_args["args"].max_steps == 10
    assert trainer_args["args"].learning_rate == 2e-4
    assert trainer_args["args"].fp16 == True
    assert trainer_args["args"].bf16 == False
    assert trainer_args["args"].logging_steps == 1
    assert trainer_args["args"].optim == "adamw_8bit"
    assert trainer_args["args"].weight_decay == 0.01
    assert trainer_args["args"].lr_scheduler_type == "linear"
    assert trainer_args["args"].seed == 3407
    assert trainer_args["args"].output_dir == "outputs"

    # Assert that the training was performed
    mock_trainer.return_value.train.assert_called_once()

    # Assert that MLflow metrics were logged correctly
    mock_log_metric.assert_called_once_with("minutesxtraining", 10.0)

    # Assert that the model was saved
    mock_model.save_pretrained_merged.assert_called_once_with(
        "model",
        mock_model.tokenizer,
        save_method="merged_16bit",
    )

    # Assert that the model was logged with MLflow
    mock_log_model.assert_called_once()
    log_model_args = mock_log_model.call_args[1]
    assert log_model_args["artifact_path"] == "phi3-instruct"
    assert isinstance(log_model_args["python_model"], Phi3)
    assert log_model_args["artifacts"] == {"snapshot": "./model"}
    assert log_model_args["registered_model_name"] == "PhiModel"


if __name__ == "__main__":
    pytest.main()
