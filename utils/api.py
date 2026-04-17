import requests
import time

BASE_URL = "https://credit-risk-api-scq0.onrender.com"


def predict_single(payload):
    url = f"{BASE_URL}/predict"

    for attempt in range(3):  # retry up to 3 times
        try:
            response = requests.post(url, json=payload, timeout=60)

            # If server gives a bad response
            if response.status_code != 200:
                return {
                    "error": True,
                    "message": f"Server error: {response.status_code}"
                }

            return response.json()

        except requests.exceptions.ReadTimeout:
            # Backend is waking up (very common on Render free tier)
            if attempt < 2:
                time.sleep(5)
            else:
                return {
                    "error": True,
                    "message": "Server is waking up... please try again in a few seconds."
                }

        except requests.exceptions.ConnectionError:
            return {
                "error": True,
                "message": "Cannot connect to backend. It might be restarting."
            }

        except Exception as e:
            return {
                "error": True,
                "message": f"Unexpected error: {str(e)}"
            }

    return {
        "error": True,
        "message": "Request failed after multiple attempts."
    }


def health_check():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        return response.json()
    except:
        return {"status": "unreachable"}