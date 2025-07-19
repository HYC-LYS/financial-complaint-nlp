import sys, joblib, pathlib

bundle = joblib.load(pathlib.Path(__file__).parent.parent /
                     "artifacts/baseline/tfidf_logreg.pkl")
model = bundle["model"]
l2p  = bundle["label2product"]

text = " ".join(sys.argv[1:]) or "I was charged unfair overdraft fees again."
pred = model.predict([text])[0]
print("l2p keys:", list(l2p.keys()))
print("pred:", pred)
print("预测类别:", l2p[pred])
