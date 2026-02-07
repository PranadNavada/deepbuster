# TextAI.py
from GPTZero.model import GPT2PPL

# Initialize GPTZero on CPU
detector = GPT2PPL(device="cpu", model_id="gpt2")

print("✅ GPTZero AI Text Detector Ready!")
print("Type 'quit' to exit.\n")

while True:
    text = input("Enter text to test: ")
    if text.lower() == "quit":
        print("Exiting...")
        break

    results, output = detector(text)
    print(f"\nResults: {results}")
    print(f"Prediction: {output}\n")
