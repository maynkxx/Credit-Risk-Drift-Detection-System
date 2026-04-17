import requests
import time

BASE_URL = "https://credit-risk-api-scq0.onrender.com"


def predict_single(payload):
    url = f"{BASE_URL}/predict"

    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=60)

            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Server error: {response.status_code}"
                }

            return response.json()

        except requests.exceptions.ReadTimeout:
            if attempt < 2:
                time.sleep(5)
            else:
                return {
                    "status": "error",
                    "message": "Server is waking up... try again in a few seconds."
                }

        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "message": "Cannot connect to backend."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    return {
        "status": "error",
        "message": "Request failed after retries"
    }


# ✅ FIXED NAME (matches frontend import)
def get_health():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ✅ NEW FUNCTION (required by model_info.py)
def get_model_info():
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=30)
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ✅ NEW FUNCTION (required by batch.py)
def predict_batch(data):
    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }