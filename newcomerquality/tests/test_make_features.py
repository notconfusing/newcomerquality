from newcomerquality.make_features import get_ores_data_dgf_from_api
from unittest.mock import patch


@patch('ores.api.Session')
def test_get_ores_data_dgf_from_api(mock_session):
    expected_response = {
      "enwiki": {
        "models": {
          "damaging": {
            "version": "0.4.0"
          },
          "goodfaith": {
            "version": "0.4.0"
          }
        },
        "scores": {
          "123": {
            "damaging": {
              "score": {
                "prediction": True,
                "probability": {
                  "false": 0.3,
                  "true": 0.7
                }
              }
            },
            "goodfaith": {
              "score": {
                "prediction": False,
                "probability": {
                  "false": 0.4,
                  "true": 0.6
                }
              }
            }
          },
          "456": {
            "damaging": {
              "score": {
                "prediction": False,
                "probability": {
                  "false": 0.5,
                  "true": 0.5
                }
              }
            },
            "goodfaith": {
              "score": {
                "prediction": True,
                "probability": {
                  "false": 0.2,
                  "true": 0.8
                }
              }
            }
          }
        }
      }
    }
    instance = mock_session.return_value
    instance.score.return_value = expected_response

    result = get_ores_data_dgf_from_api([123, 456], 'enwiki')
    assert expected_response == result
