import sys
import pytest
from unittest.mock import MagicMock
from types import ModuleType


class MockModule(ModuleType):
    """A custom module class that creates attributes on-the-fly and allows assignment."""

    def __init__(self, name):
        super().__init__(name)
        self._mock_dict = {}

    def __getattr__(self, name):
        if name not in self._mock_dict:
            self._mock_dict[name] = MockModule(f"{self.__name__}.{name}")
        return self._mock_dict[name]

    def __setattr__(self, name, value):
        if name == "_mock_dict":
            super().__setattr__(name, value)
        else:
            self._mock_dict[name] = value


def create_mock_mlflow():
    mock_mlflow = MockModule("mlflow")

    # Mock specific functions
    mock_mlflow.start_run = MagicMock()
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.log_param = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    mock_mlflow.pyfunc.log_model = MagicMock()

    # Mock classes
    mock_mlflow.models.signature.ModelSignature = type("ModelSignature", (), {})
    mock_mlflow.types.ColSpec = type("ColSpec", (), {})
    mock_mlflow.types.DataType = type("DataType", (), {})
    mock_mlflow.types.ParamSchema = type("ParamSchema", (), {})
    mock_mlflow.types.ParamSpec = type("ParamSpec", (), {})
    mock_mlflow.types.Schema = type("Schema", (), {})

    return mock_mlflow

    # def create_autospec_mock():
    #    """Create a MagicMock that creates new mocks for accessed attributes."""
    #    return MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
    #
    #
    # def mock_module(module_name, module_dict):
    #    """Recursively mock a module and its submodules."""
    #    mock = ModuleType(module_name)
    #    for attr, value in module_dict.items():
    #        if isinstance(value, dict):
    #            setattr(mock, attr, mock_module(f"{module_name}.{attr}", value))
    #        else:
    #            setattr(mock, attr, create_autospec_mock())
    #    return mock


@pytest.fixture(autouse=True)
def mock_imports(monkeypatch):
    mock_mlflow = create_mock_mlflow()

    mock_FastLanguageModel = MagicMock()
    mock_unsloth = MagicMock()
    mock_FastLanguageModel.from_pretrained.return_value = (MagicMock(), MagicMock())
    mock_unsloth.FastLanguageModel = mock_FastLanguageModel

    mocks = {
        "mlflow": mock_mlflow,
        "unsloth": mock_unsloth,
        "torch": MockModule("torch"),
        "datasets": MockModule("datasets"),
        "trl": MockModule("trl"),
        "transformers": MockModule("transformers"),
        "fastapi": MockModule("fastapi"),
    }

    for name, mock in mocks.items():
        monkeypatch.setitem(sys.modules, name, mock)

    return mock_mlflow


def test_train_my_model_mlflow_logging(mock_imports):
    mlflow = mock_imports  # This is now the mocked mlflow module

    # Import the function to be tested
    from trainmodel import train_my_model

    # Mock FastLanguageModel, is_bfloat16_supported, TrainingArguments, and SFTTrainer
    with pytest.MonkeyPatch().context() as m:
        # Mock FastLanguageModel.from_pretrained to return a tuple (model, tokenizer)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_from_pretrained = MagicMock(return_value=(mock_model, mock_tokenizer))
        m.setattr("trainmodel.FastLanguageModel.from_pretrained", mock_from_pretrained)

        m.setattr("trainmodel.is_bfloat16_supported", lambda: False)
        m.setattr("trainmodel.TrainingArguments", MagicMock())

        # Mock SFTTrainer
        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_runtime": 600}  # 10 minutes
        mock_trainer.train.return_value = mock_train_result
        mock_sft_trainer = MagicMock(return_value=mock_trainer)
        m.setattr("trainmodel.SFTTrainer", mock_sft_trainer)

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

    # Verify that FastLanguageModel.from_pretrained was called
    mock_from_pretrained.assert_called_once()

    # Verify that SFTTrainer was initialized with the correct model and tokenizer
    mock_sft_trainer.assert_called_once()
    _, kwargs = mock_sft_trainer.call_args
    assert kwargs["model"] == mock_model
    assert kwargs["tokenizer"] == mock_tokenizer
