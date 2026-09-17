import re
import json

def parse_training_log(text):
    results = []

    # Match each epoch block
    pattern = re.compile(
        r"========== Epoch\s+(?P<epoch>\d+)\s+=========\s*"
        r"Train loss:\s*(?P<train_loss>-?\d+\.\d+):\s*"
        r"NLL on val set:\s*(?P<val_nll>-?\d+\.\d+),\s*"
        r"Posterior NLL on val:\s*(?P<posterior_val_nll>-?\d+\.\d+)\s*"
        r"Posterior:\s*NLL\s*(?P<posterior_nll>-?\d+\.\d+):\s*"
        r"kl_loss\s*(?P<posterior_kl>-?\d+\.\d+):\s*"
        r"Prior:\s*NLL\s*(?P<prior_nll>-?\d+\.\d+):\s*"
        r"kl_loss\s*(?P<prior_kl>-?\d+\.\d+):\s*"
        r"Layered KL:\s*tensor\(\[(?P<layered_kl>[^\]]+)\]",
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(text):
        d = match.groupdict()

        results.append({
            "epoch": int(d["epoch"]),
            "train_loss": float(d["train_loss"]),
            "val_nll": float(d["val_nll"]),
            "posterior_val_nll": float(d["posterior_val_nll"]),
            "posterior": {
                "nll": float(d["posterior_nll"]),
                "kl_loss": float(d["posterior_kl"])
            },
            "prior": {
                "nll": float(d["prior_nll"]),
                "kl_loss": float(d["prior_kl"])
            },
            "layered_kl": [
                float(x.strip())
                for x in d["layered_kl"].split(",")
            ]
        })

    return results


# Example usage
with open("untitled.txt", "r") as f:
    text = f.read()

parsed = parse_training_log(text)

# Write JSON
with open("training.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(json.dumps(parsed[:2], indent=2))
