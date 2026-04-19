import io
import types

import numpy as np

from src.app import utils


class FakeEncoder:
	def __init__(self, classes):
		self.classes_ = np.array(classes)

	def transform(self, values):
		mapping = {v: i for i, v in enumerate(self.classes_)}
		return np.array([mapping[v] for v in values])


def test_prepare_prediction_input_encodes_and_defaults_expected_behavior(field):
	form = types.SimpleNamespace(
		child_age=field("11"),
		child_gender=field("Male"),
		child_ethnicity=field("Unknown Ethnicity"),
		child_prior_placements=field("2"),
		returning_child=field("true"),
		missing_episodes=field("1"),
		sibling_group_size=field("0"),
		placed_with_siblings=field("false"),
		carer_age=field("40"),
		carer_gender=field("Female"),
		carer_ethnicity=field("Unknown"),
		eh_involvement=field("yes"),
		yot_involvement=field("no"),
	)
	encoders = {
		"Child Gender": FakeEncoder(["Unknown", "Male", "Female"]),
		"Child Ethnicity": FakeEncoder(["Unknown", "White - British"]),
		"Carer Gender": FakeEncoder(["Unknown", "Male", "Female"]),
		"Carer Ethnicity": FakeEncoder(["Unknown", "White - British"]),
	}
	arr = utils.prepare_prediction_input(form, encoders)
	assert arr.shape == (1, len(utils.REGRESSION_FEATURE_COLUMNS))
	assert arr[0][3] == 2
	assert arr[0][13] == 3


def test_derive_explanation_factors_returns_top_n_expected_behavior():
	factors = utils._derive_explanation_factors([1, 2, 3], ["a", "b", "c"], [0.1, 0.9, 0.3], max_items=2)
	assert len(factors) == 2
	assert factors[0].startswith("b")


def test_generate_predictions_list_handles_missing_models_expected_behavior():
	out = utils.generate_predictions_list(np.array([[1, 2, 3]]), None, None, None)
	assert out[0]["type"] == "Error"


def test_generate_predictions_list_returns_ranked_predictions_expected_behavior():
	class RF:
		classes_ = np.array([0, 1])
		feature_importances_ = np.array([0.5, 0.5])

		def predict_proba(self, _x):
			return np.array([[0.2, 0.8]])

	class LR:
		def predict(self, _x):
			return np.array([200])

	class Breakdown:
		classes_ = np.array([0, 1])
		feature_importances_ = np.array([0.4, 0.6, 0.3])

		def predict_proba(self, _x):
			return np.array([[0.7, 0.3]])

	placement_encoder = FakeEncoder(["Kinship", "External Fostering"])
	input_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 40, 1, 1, 0, 0, 1, 0]])
	preds = utils.generate_predictions_list(
		input_data,
		RF(),
		LR(),
		placement_encoder,
		breakdown_model=Breakdown(),
		placement_feature_names=["f1", "f2"],
		breakdown_feature_names=["f1", "f2", "f3"],
	)
	assert len(preds) == 2
	assert "net_stability" in preds[0]


def test_extract_profile_from_form_maps_fields_expected_behavior(field):
	form = types.SimpleNamespace(
		child_age=field(10),
		child_gender=field("Male"),
		child_ethnicity=field("White - British"),
		child_prior_placements=field(0),
		returning_child=field("False"),
		missing_episodes=field(0),
		sibling_group_size=field(0),
		placed_with_siblings=field("False"),
		carer_age=field(45),
		carer_gender=field("Female"),
		carer_ethnicity=field("White - British"),
		eh_involvement=field("False"),
		yot_involvement=field("False"),
	)
	profile = utils.extract_profile_from_form(form)
	assert profile["carer_age"] == 45


def test_compare_placement_options_filters_selected_types_expected_behavior(monkeypatch):
	monkeypatch.setattr(utils, "prepare_prediction_input", lambda *_args, **_kwargs: np.array([[1] * 15]))
	monkeypatch.setattr(
		utils,
		"generate_predictions_list",
		lambda *_args, **_kwargs: [{"type": "Kinship"}, {"type": "External Fostering"}],
	)
	out = utils.compare_placement_options({}, ["Kinship"], object(), object(), {}, object())
	assert out == [{"type": "Kinship"}]


def test_process_bulk_upload_success_and_failure_expected_behavior(monkeypatch):
	captured = []

	def fake_add(_conn, placement_data):
		captured.append(placement_data)

	monkeypatch.setattr("src.app.models.add_placement_record", fake_add)

	headers = ",".join(utils.BCFT_UPLOAD_SCHEMA)
	row_ok = "10,Male,White - British,1,False,0,0,False,35,Female,White - British,Kinship,2024-01-01,2024-02-01,30,planned move,5,False,False,2"
	csv_bytes = (headers + "\n" + row_ok).encode("utf-8")
	results = utils.process_bulk_upload(object(), io.BytesIO(csv_bytes), user_id=1)
	assert results["success"] == 1
	assert results["failed"] == 0
	assert captured[0]["uploaded_by"] == 1


def test_process_bulk_upload_missing_schema_fails_expected_behavior():
	bad_csv = b"wrong,headers\n1,2"
	results = utils.process_bulk_upload(object(), io.BytesIO(bad_csv), user_id=1)
	assert results["failed"] >= 1
	assert any("schema mismatch" in err.lower() for err in results["errors"])

