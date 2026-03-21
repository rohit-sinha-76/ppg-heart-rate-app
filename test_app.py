import pytest
from app import app
from unittest.mock import patch
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test standard layout Dashboard rendering loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"PPG Predictor" in response.data

@patch('app.model.predict')
def test_predict_success(mock_predict, client):
    """Test successful heart rate prediction with correct input size layout."""
    mock_predict.return_value = np.array([[72.53]])
    mock_signal = [0.1] * 1000  # Strictly 1000 duration window
    
    response = client.post('/predict', json={"ppg_signal": mock_signal})
    assert response.status_code == 200
    
    data = response.get_json()
    assert "heart_rate" in data
    assert data["heart_rate"] == 72.5  # Rounded 1 decimal in app.py

def test_predict_validation_failure(client):
    """Test predictive block Trigger handles length alignment correctly."""
    mock_signal = [0.1] * 500  # Invalid length
    
    response = client.post('/predict', json={"ppg_signal": mock_signal})
    assert response.status_code == 400
    
    data = response.get_json()
    assert "error" in data
    assert "Invalid signal length" in data["error"]
