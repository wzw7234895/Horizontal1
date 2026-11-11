from types import SimpleNamespace

from person_tracking.experiments.train_person_static_v1 import normalize_result_metrics
from person_tracking.experiments.train_person_v2 import resolve_initialization


def test_normalize_result_metrics_accepts_ultralytics_metric_object():
    result = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.716, "fitness": 0.304})

    assert normalize_result_metrics(result) == {
        "metrics/mAP50(B)": 0.716,
        "fitness": 0.304,
    }


def test_normalize_result_metrics_accepts_plain_mapping():
    assert normalize_result_metrics({"fitness": 0.25, "ignored": None}) == {"fitness": 0.25}


def test_explicit_checkpoint_does_not_use_tracking_model(tmp_path, monkeypatch):
    import person_tracking.experiments.train_person_v2 as training
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_bytes(b"local test checkpoint placeholder")

    def unexpected_configuration_read():
        raise AssertionError("Explicit weights must not read the tracking model")

    monkeypatch.setattr(training, "load_config", unexpected_configuration_read)
    assert resolve_initialization(checkpoint) == checkpoint.resolve()
