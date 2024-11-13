import requests
import json

def get_medical_advice(symptoms):
    """
    Uses the Gemini API to get medical advice based on symptoms.

    Args:
        symptoms (str): A string describing the symptoms.

    Returns:
        str: A suggested type of doctor or specialist.
    """

    url = "https://api.gemini.com/chat"
    headers = {"Authorization": "AIzaSyAR2AFJcW8b8VSodqz1vv5khbPSHEQWGBg"}  # Replace with your actual API key
    data = {
        "prompt": f"I am experiencing {symptoms}. What type of doctor should I see?",
        "model_name": "gemini-1.5-flash"
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()

        # Check for the 'response' key or other relevant key based on documentation
        if 'response' in response_json:
            suggested_doctor = response_json['response']
        elif 'recommendation' in response_json:  # Assuming "recommendation" is the new key
            suggested_doctor = response_json['recommendation']
        else:
            print("Unexpected response format. Please consult the API documentation.")
            return None

        return suggested_doctor

    except Exception as e:
        print("Error:", e)
        return None  # Indicate error

if __name__ == "__main__":
    symptoms = input("Please describe your symptoms: ")
    suggested_doctor = get_medical_advice(symptoms)

    if suggested_doctor:
        print(f"Based on your symptoms, you might consider consulting a {suggested_doctor}.")
    else:
        print("An error occurred. Please try again or consult a healthcare professional.")
