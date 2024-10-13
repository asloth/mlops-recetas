import pytest
from unittest.mock import patch, MagicMock


# Mock all the imports
@pytest.fixture(autouse=True)
def mock_imports():
    mock_unsloth = MagicMock()
    mock_FastLanguageModel = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_unsloth.FastLanguageModel = mock_FastLanguageModel
    mocks = {
        "unsloth": mock_unsloth,
        "torch": MagicMock(),
        "datasets": MagicMock(),
        "trl": MagicMock(),
        "transformers": MagicMock(),
        "fastapi": MagicMock(),
        "mlflow": MagicMock(),
    }
    with patch.dict("sys.modules", mocks):
        yield mocks["mlflow"]


def test_train_my_model_mlflow_logging(mock_imports):
    mlflow = mock_imports  # This is now the mocked mlflow module

    # Import the function to be tested
    from trainmodel import train_my_model

    # Set up MLflow mock methods
    mlflow.start_run = MagicMock()
    mlflow.set_tracking_uri = MagicMock()
    mlflow.set_experiment = MagicMock()
    mlflow.log_param = MagicMock()
    mlflow.log_metric = MagicMock()

    # Ensure it returns two values as expected
    # Mock FastLanguageModel, is_bfloat16_supported, TrainingArguments, and SFTTrainer
    with patch("trainmodel.FastLanguageModel", return_value=MagicMock()), patch(
        "trainmodel.is_bfloat16_supported", return_value=False
    ), patch("trainmodel.TrainingArguments", MagicMock()), patch(
        "trainmodel.SFTTrainer", MagicMock()
    ):

        # Mock the trainer.train() method to return a MagicMock with metrics
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_runtime": 600}  # 10 minutes
        with patch("trainmodel.SFTTrainer.train", return_value=mock_train_result):

            # Call the function
            train_my_model()

    # Verify MLflow logging calls
    mlflow.set_tracking_uri.assert_called_once_with("http://mlflow-server:5000")
    mlflow.set_experiment.assert_called_once_with("recetas-model")

    expected_params = {
        "warmup_steps": 5,
        "per_device_train_batch_size": 2,
        "max_steps": 10,
        "weight_decay": 0.01,
    }
    for param, value in expected_params.items():
        mlflow.log_param.assert_any_call(param, value)

    mlflow.log_metric.assert_called_once_with("minutesxtraining", 10.0)
