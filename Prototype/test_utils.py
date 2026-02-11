import unittest
import json
import os
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock
from utils import PlacementProfile, load_profiles, save_profiles, generate_predictions_list


class TestPlacementProfile(unittest.TestCase):

    def setUp(self):
        """Set up test data before each test."""
        self.test_attributes = {
            "name": "Test Profile",
            "childAge": 10,
            "childGender": "Male",
            "childEthnicity": "White - British",
            "childPriorPlacementsNum": 2,
            "returningChild": "No",
            "ageChildLeftCare": "N/A",
            "carerAge": 45,
            "missingEpisodes": 0,
            "numberOfCarers": 2,
            "carerGender": "Female",
            "placedWithSiblings": "Yes",
            "siblingGroupSize": 2,
            "ageChildCameIntoCare": 8,
            "carerEthnicity": "White - British",
            "involvementOfEH": "No",
            "siblingsInEH": "No",
            "involvementOfYOT": "No"
        }

    def test_profile_initialization(self):
        """Test that a profile is initialized correctly."""
        profile = PlacementProfile("Test", self.test_attributes)
        self.assertEqual(profile.get_name(), "Test")
        self.assertEqual(profile.get_attribute("childAge"), 10)
        self.assertEqual(profile.get_attribute("childGender"), "Male")

    def test_to_dict(self):
        """Test conversion of profile to dictionary."""
        profile = PlacementProfile("Test", self.test_attributes)
        result = profile.to_dict()
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["childAge"], 10)
        self.assertIsInstance(result, dict)

    def test_edit_name(self):
        """Test editing profile name."""
        profile = PlacementProfile("Old Name", self.test_attributes)
        profile.edit_name("New Name")
        self.assertEqual(profile.get_name(), "New Name")

    def test_edit_attribute(self):
        """Test editing a specific attribute."""
        profile = PlacementProfile("Test", self.test_attributes)
        profile.edit_attribute("childAge", 12)
        self.assertEqual(profile.get_attribute("childAge"), 12)

    def test_edit_invalid_attribute_raises_error(self):
        """Test that editing an invalid attribute raises AttributeError."""
        profile = PlacementProfile("Test", self.test_attributes)
        with self.assertRaises(AttributeError):
            profile.edit_attribute("invalidAttribute", "value")

    def test_get_invalid_attribute_raises_error(self):
        """Test that getting an invalid attribute raises AttributeError."""
        profile = PlacementProfile("Test", self.test_attributes)
        with self.assertRaises(AttributeError):
            profile.get_attribute("invalidAttribute")


class TestProfileFileOperations(unittest.TestCase):

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file_path = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        """Remove temporary file after testing."""
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def test_save_and_load_profiles(self):
        """Test saving and loading profiles."""
        profiles = [
            PlacementProfile("Profile1", {"childAge": 10, "childGender": "Male"}),
            PlacementProfile("Profile2", {"childAge": 12, "childGender": "Female"})
        ]

        save_profiles(self.temp_file_path, profiles)
        loaded_profiles = load_profiles(self.temp_file_path)

        self.assertEqual(len(loaded_profiles), 2)
        self.assertEqual(loaded_profiles[0].get_name(), "Profile1")
        self.assertEqual(loaded_profiles[1].get_name(), "Profile2")

    def test_load_profiles_file_not_found(self):
        """Test loading profiles when file doesn't exist."""
        profiles = load_profiles("nonexistent_file.txt")
        self.assertEqual(profiles, [])


class TestGeneratePredictionsList(unittest.TestCase):

    def setUp(self):
        """Set up mock models and encoder."""
        self.mock_rf_model = Mock()
        self.mock_lr_model = Mock()
        self.mock_placement_encoder = Mock()

        # Mock RF classifier to return probabilities for 4 classes
        self.mock_rf_model.predict_proba.return_value = np.array([[0.1, 0.3, 0.5, 0.1]])

        # Mock LR regressor to return different durations
        self.mock_lr_model.predict.side_effect = [100, 200, 300, 400]

        # Mock encoder to return placement type names
        self.mock_placement_encoder.inverse_transform.side_effect = [
            ["Fostering - Long Term"],
            ["Kinship"],
            ["Residential"],
            ["Adoption"]
        ]

    def test_generate_predictions_list(self):
        """Test that predictions list is generated correctly."""
        input_data = [10, 1, 2, 45, 3, 4, 0]

        predictions = generate_predictions_list(
            input_data,
            self.mock_rf_model,
            self.mock_lr_model,
            self.mock_placement_encoder
        )

        self.assertEqual(len(predictions), 4)
        self.assertIsInstance(predictions, list)
        self.assertTrue(all(isinstance(p, dict) for p in predictions))
        self.assertTrue(all("type" in p and "days" in p for p in predictions))

    def test_predictions_sorted_by_days(self):
        """Test that predictions are sorted by duration ascending."""
        input_data = [10, 1, 2, 45, 3, 4, 0]

        predictions = generate_predictions_list(
            input_data,
            self.mock_rf_model,
            self.mock_lr_model,
            self.mock_placement_encoder
        )

        days_list = [p["days"] for p in predictions]
        self.assertEqual(days_list, sorted(days_list))


if __name__ == '__main__':
    unittest.main()