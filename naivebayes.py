# ============================================================
#  Naïve Bayes Classifier — Will the Student Study?
#  Implements the same steps shown in class:
#    1. Build frequency / probability tables
#    2. Apply P(y|X) ∝ P(y) × ∏ P(xᵢ|y)
#    3. Predict using argmax
# ============================================================

import pandas as pd
from fractions import Fraction

# ── 1. Dataset ───────────────────────────────────────────────
data = {
    "Mood":        ["Happy",   "Happy",   "Stressed","Neutral", "Happy",
                    "Stressed","Neutral", "Stressed","Happy",   "Neutral",
                    "Happy",   "Stressed","Neutral", "Happy"],
    "Energy":      ["High",    "High",    "Low",     "Medium",  "Medium",
                    "Low",     "High",    "Medium",  "High",    "Low",
                    "Medium",  "High",    "Medium",  "Low"],
    "Distraction": ["Low",     "High",    "High",    "Low",     "High",
                    "Low",     "Low",     "High",    "Low",     "High",
                    "Low",     "High",    "High",    "High"],
    "Deadline":    ["Near",    "Near",    "Far",     "Near",    "Far",
                    "Near",    "Near",    "Far",     "Far",     "Near",
                    "Near",    "Far",     "Near",    "Far"],
    "Study":       ["Yes",     "Yes",     "No",      "Yes",     "No",
                    "Yes",     "Yes",     "No",      "Yes",     "Yes",
                    "Yes",     "No",      "Yes",     "No"],
}

df = pd.DataFrame(data)
features = ["Mood", "Energy", "Distraction", "Deadline"]
target   = "Study"
classes  = ["Yes", "No"]

print("=" * 60)
print("  DATASET")
print("=" * 60)
print(df.to_string(index=True))
print()


# ── 2. Data Preparation — Frequency & Probability Tables ─────
print("=" * 60)
print("  DATA PREPARATION — FREQUENCY & PROBABILITY TABLES")
print("=" * 60)

# Prior counts
total      = len(df)
class_counts = df[target].value_counts()

print(f"\n── Prior (Study) ──")
print(f"{'Study':<8} {'Count':>6}   {'P(Yes)/P(No)':>14}")
print("-" * 32)
for cls in classes:
    cnt = class_counts[cls]
    frac = Fraction(cnt, total)
    print(f"{cls:<8} {cnt:>6}   {str(frac):>14}")
print(f"{'Total':<8} {total:>6}   {'100%':>14}")

# Conditional frequency tables per feature
cond_probs = {}   # cond_probs[feature][value][class] = probability (float)

for feat in features:
    print(f"\n── {feat} ──")
    vals = df[feat].unique()
    header = f"{'':12} {'Yes':>6} {'No':>6}   {'P(yes)':>8} {'P(no)':>8}"
    print(header)
    print("-" * len(header))

    cond_probs[feat] = {}
    for val in sorted(vals):
        cond_probs[feat][val] = {}
        row_counts = {}
        row_probs  = {}
        for cls in classes:
            cnt = len(df[(df[feat] == val) & (df[target] == cls)])
            row_counts[cls] = cnt
            row_probs[cls]  = cnt / class_counts[cls]
            cond_probs[feat][val][cls] = row_probs[cls]

        frac_yes = Fraction(row_counts["Yes"], class_counts["Yes"])
        frac_no  = Fraction(row_counts["No"],  class_counts["No"])
        print(f"{val:<12} {row_counts['Yes']:>6} {row_counts['No']:>6}"
              f"   {str(frac_yes):>8} {str(frac_no):>8}")

    # Totals
    print(f"{'Total':<12} {class_counts['Yes']:>6} {class_counts['No']:>6}"
          f"   {'100%':>8} {'100%':>8}")


# ── 3. Prediction Function ────────────────────────────────────
def naive_bayes_predict(sample: dict, verbose: bool = True) -> str:
    """
    Classify a sample using Naïve Bayes.
    sample = {feature: value, ...}
    Returns predicted class label.
    """
    scores = {}

    if verbose:
        print("\n" + "=" * 60)
        print("  APPLYING NAÏVE BAYES FORMULA")
        print("=" * 60)
        print(f"\n  Sample: {sample}\n")
        print("  P(y|X) ∝ P(y) × ∏ P(xᵢ|y)\n")

    for cls in classes:
        prior = class_counts[cls] / total
        product = prior
        breakdown = [f"P({cls}) = {Fraction(class_counts[cls], total)}"]

        for feat, val in sample.items():
            p = cond_probs[feat][val][cls]
            cnt = int(p * class_counts[cls])
            frac = Fraction(cnt, class_counts[cls])
            breakdown.append(f"P({val}|{cls}) = {frac}")
            product *= p

        scores[cls] = product

        if verbose:
            print(f"  P({cls}|X) = " + " × ".join(breakdown))
            print(f"           = {product:.6f}\n")

    predicted = max(scores, key=scores.get)

    if verbose:
        print("-" * 60)
        print(f"  P(Yes|X) = {scores['Yes']:.6f}")
        print(f"  P(No|X)  = {scores['No']:.6f}")
        print()
        winner = "P(Yes|X)" if scores["Yes"] > scores["No"] else "P(No|X)"
        print(f"  {winner} is greater  →  argmax = {predicted}")
        print("=" * 60)
        print(f"\n  PREDICTION: The student will {predicted.upper()} study.")
        print("=" * 60)

    return predicted


# ── 4. Solve the Sample Problem ───────────────────────────────
sample_input = {
    "Mood":        "Neutral",
    "Energy":      "Medium",
    "Distraction": "Low",
    "Deadline":    "Far",
}

result = naive_bayes_predict(sample_input, verbose=True)


# ── 5. Test Multiple Samples ──────────────────────────────────
print("\n\n" + "=" * 60)
print("  BATCH PREDICTIONS")
print("=" * 60)

test_cases = [
    {"Mood": "Neutral",  "Energy": "Medium", "Distraction": "Low",  "Deadline": "Far"},
    {"Mood": "Happy",    "Energy": "High",   "Distraction": "Low",  "Deadline": "Near"},
    {"Mood": "Stressed", "Energy": "Low",    "Distraction": "High", "Deadline": "Far"},
    {"Mood": "Neutral",  "Energy": "Low",    "Distraction": "High", "Deadline": "Far"},
    {"Mood": "Happy",    "Energy": "Medium", "Distraction": "High", "Deadline": "Near"},
]

print(f"\n{'Mood':<12} {'Energy':<10} {'Distraction':<14} {'Deadline':<10} {'Prediction':>12}")
print("-" * 62)
for tc in test_cases:
    pred = naive_bayes_predict(tc, verbose=False)
    symbol = "✓ Yes" if pred == "Yes" else "✗ No"
    print(f"{tc['Mood']:<12} {tc['Energy']:<10} {tc['Distraction']:<14} {tc['Deadline']:<10} {symbol:>12}")