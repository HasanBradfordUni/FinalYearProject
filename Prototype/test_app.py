import unittest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from app import app, preprocess_input


class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        """Set up test client and test data."""
        app.config['TESTING'] = True
        self.client = app.test_client()

        # Create temporary profiles file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file_path = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def test_index_route(self):
        """Test that the index page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)

    @patch('app.generate_predictions_list')
    def test_predict_route(self, mock_predictions):
        """Test the predict route."""
        mock_predictions.return_value = [
            {"type": "Fostering - Long Term", "days": 100},
            {"type": "Kinship", "days": 200},
            {"type": "Residential", "days": 300},
            {"type": "Adoption", "days": 400}
        ]

        form_data = {
            "childAge": "10",
            "childGender": "Male",
            "childEthnicity": "White - British",
            "carerAge": "45",
            "carerGender": "Female",
            "carerEthnicity": "White - British"
        }

        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fostering - Long Term', response.data)

    @patch('app.save_profiles')
    @patch('app.placement_profiles', [])
    def test_add_profile_route(self, mock_save):
        """Test adding a new profile."""
        form_data = {
            "profileName": "Test Profile",
            "childAge": "10",
            "childGender": "Male",
            "childEthnicity": "White - British",
            "carerAge": "45",
            "carerGender": "Female",
            "carerEthnicity": "White - British",
            "childPriorPlacementsNum": "2",
            "returningChild": "No",
            "ageChildLeftCare": "15",
            "missingEpisodes": "0",
            "numberOfCarers": "2",
            "placedWithSiblings": "Yes",
            "siblingGroupSize": "2",
            "ageChildCameIntoCare": "8",
            "involvementOfEH": "No",
            "siblingsInEH": "No",
            "involvementOfYOT": "No"
        }

        response = self.client.post('/add_profile', data=form_data)
        self.assertEqual(response.status_code, 302)  # Redirect
        mock_save.assert_called_once()

    @patch('app.generate_predictions_list')
    @patch('app.placement_profiles')
    def test_compare_route(self, mock_profiles, mock_predictions):
        """Test the compare route."""
        # Mock a profile
        mock_profile = Mock()
        mock_profile.get_name.return_value = "Test Profile"
        mock_profile.to_dict.return_value = {
            "name": "Test Profile",
            "childAge": 10,
            "childGender": "Male",
            "childEthnicity": "White - British",
            "carerAge": 45,
            "carerGender": "Female",
            "carerEthnicity": "White - British"
        }
        mock_profiles.__iter__.return_value = [mock_profile]

        mock_predictions.return_value = [
            {"type": "Kinship", "days": 150},
            {"type": "Fostering - Long Term", "days": 250}
        ]

        form_data = {
            "profileName": "Test Profile",
            "placementTypes": ["Kinship", "Fostering - Long Term"]
        }

        response = self.client.post('/compare', data=form_data)
        self.assertEqual(response.status_code, 200)

    def test_compare_route_no_profile(self):
        """Test compare route when no profile is selected."""
        form_data = {
            "placementTypes": ["Kinship", "Fostering - Long Term"]
        }

        response = self.client.post('/compare', data=form_data)
        self.assertIn(b'Error: No profile selected', response.data)

    def test_compare_route_invalid_placement_count(self):
        """Test compare route with invalid number of placement types."""
        form_data = {
            "profileName": "Test Profile",
            "placementTypes": ["Kinship"]  # Only 1, need 2-4
        }

        response = self.client.post('/compare', data=form_data)
        self.assertIn(b'Please select between 2 and 4', response.data)


class TestPreprocessInput(unittest.TestCase):

    @patch('app.feature_encoders')
    def test_preprocess_input(self, mock_encoders):
        """Test input preprocessing."""
        # Mock encoders
        mock_encoder = Mock()
        mock_encoder.transform.return_value = [0]
        mock_encoders.items.return_value = [
            ("Child Gender", mock_encoder),
            ("Child Ethnicity", mock_encoder),
            ("Carer Gender Composition", mock_encoder),
            ("Carer Ethnicity Or Religion", mock_encoder)
        ]

        form_data = {
            "childAge": "10",
            "childGender": "Male",
            "childEthnicity": "White - British",
            "carerAge": "45",
            "carerGender": "Female",
            "carerEthnicity": "White - British"
        }

        result = preprocess_input(form_data)
        self.assertEqual(result.shape[1], 7)  # 7 features
        self.assertEqual(result[0][-1], 0)  # Placement Type placeholder


if __name__ == '__main__':
    unittest.main()