import requests
import json

def send_prediction_request(url, state):
    # Server URL
    server_url = 'http://localhost:5000/predict'  # Adjust if your server is running on a different address

    # Prepare the data
    data = {
        'url': url,
        'state': state
    }

    # Send POST request
    response = requests.post(server_url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage
image_url = "ceshitu/1-1.jpg"  # Replace with your image URL
state = "2"  # Replace with the desired state (1-9)

result = send_prediction_request(image_url, state)

if result:
    print("Prediction result:")
    print(json.dumps(result, indent=2))