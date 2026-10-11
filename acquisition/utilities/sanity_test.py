"""
    Experiment sanity checks: motor-movement watchdog and optional SMS alerts.

    SMS alerts use Twilio and are optional. Credentials are read from
    environment variables and are never stored in the code:
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
    and the recipient from ALERT_PHONE_NUMBER (see the experiment notebook).

    Author: Monica Coraggioso
"""
import os
import math


def send_sms(to_number, body):
    """Send an SMS alert via Twilio. Silently skipped if Twilio is not configured."""
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER")
    if not (account_sid and auth_token and from_number and to_number):
        print("SMS alert skipped: Twilio environment variables or recipient not set.")
        return

    from twilio.rest import Client  # optional dependency: pip install twilio
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(body=body, from_=from_number, to=to_number)
        print(f"Message sent successfully: {message.sid}")
    except Exception as e:
        print(f"Error sending message: {e}")

def calculate_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to check if the motor has moved more than 1 cm in the last 2 minutes
def check_motor_movement(motor_positions, last_points):
    # Get the last 20*60*2=2400 positions (if available) ->  ~2min
    recent_positions = motor_positions[-last_points:]

    # Calculate the distance moved between the first and last position
    start_position = recent_positions[0]  # Extract position
    end_position = recent_positions[-1]  # Extract position
    distance_moved = calculate_distance(start_position, end_position)

    # Check if the distance moved is greater than 1 cm
    if distance_moved > 10:
        return True  # Motor moved more than 1 cm
    else:
        return False  # Motor did not move more than 1 cm
