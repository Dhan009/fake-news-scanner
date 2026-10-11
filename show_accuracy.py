import pickle

with open("saved_models/accuracies.pkl", "rb") as f:
    results = pickle.load(f)

print("\nModel accuracy on the test set:")
for name, acc in results.items():
    print(f"  {name:20s} {acc * 100:.2f}%")