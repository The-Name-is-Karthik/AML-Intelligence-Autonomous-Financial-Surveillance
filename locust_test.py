from locust import HttpUser, task, between

class AMLUser(HttpUser):
    wait_time = between(0.1, 0.5) # Simulate rapid-fire transactions

    @task
    def test_predict(self):
        payload = {
            "Time": "14:30:05", "Date": "2026-02-12",
            "Sender_account": 998877, "Receiver_account": 112233,
            "Amount": 7500.0, "Payment_currency": "INR", "Received_currency": "INR",
            "Sender_bank_location": "India", "Receiver_bank_location": "Turkey",
            "Payment_type": "Transfer"
        }
        self.client.post("/predict", json=payload)