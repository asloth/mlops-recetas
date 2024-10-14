import pytest
import sys
from unittest.mock import MagicMock
from types import ModuleType


def create_autospec_mock():
    """Create a MagicMock that creates new mocks for accessed attributes."""
    return MagicMock(side_effect=lambda *args, **kwargs: MagicMock())


def mock_module(module_name, module_dict):
    """Recursively mock a module and its submodules."""
    mock = ModuleType(module_name)
    for attr, value in module_dict.items():
        if isinstance(value, dict):
            setattr(mock, attr, mock_module(f"{module_name}.{attr}", value))
        else:
            setattr(mock, attr, create_autospec_mock())
    return mock


@pytest.fixture(autouse=True)
def mock_imports(monkeypatch):
    mlflow_mock = mock_module(
        "mlflow",
        {
            "start_run": None,
            "set_tracking_uri": None,
            "set_experiment": None,
            "log_param": None,
            "log_metric": None,
            "pyfunc": {
                "log_model": None,
            },
            "models": {
                "signature": {
                    "ModelSignature": None,
                },
            },
        },
    )

    mocks = {
        "mlflow": mlflow_mock,
        "unsloth": create_autospec_mock(),
        "torch": create_autospec_mock(),
        "datasets": create_autospec_mock(),
        "trl": create_autospec_mock(),
        "transformers": create_autospec_mock(),
        "fastapi": create_autospec_mock(),
    }

    for name, mock in mocks.items():
        monkeypatch.setitem(sys.modules, name, mock)

    return mocks["mlflow"]


def test_train_my_model_mlflow_logging(mock_imports):
    mlflow = mock_imports  # This is now the mocked mlflow module

    # Import the function to be tested
    from trainmodel import train_my_model

    # Mock FastLanguageModel, is_bfloat16_supported, TrainingArguments, and SFTTrainer
    with pytest.MonkeyPatch().context() as m:
        m.setattr("trainmodel.FastLanguageModel", create_autospec_mock())
        m.setattr("trainmodel.is_bfloat16_supported", lambda: False)
        m.setattr("trainmodel.TrainingArguments", create_autospec_mock())
        m.setattr("trainmodel.SFTTrainer", create_autospec_mock())

        # Mock the trainer.train() method to return a MagicMock with metrics
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_runtime": 600}  # 10 minutes
        m.setattr("trainmodel.SFTTrainer.train", lambda self: mock_train_result)

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

    # Check if mlflow.pyfunc.log_model was called
    mlflow.pyfunc.log_model.assert_called_once()
