from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_selection import rank_genes_by_ttest


DATA_PATH = Path("data/processed/golub_full.csv")
RANKING_PATH = Path("data/processed/gene_ranking.csv")
K_VALUES = [10, 25, 50, 100, 150, 200]


def make_pipeline():
	return Pipeline(
		[
			("scaler", StandardScaler()),
			("svc", SVC(kernel="linear", probability=True, random_state=42)),
		]
	)


def main():
	df = pd.read_csv(DATA_PATH)
	ranked_genes = rank_genes_by_ttest(df)
	pd.DataFrame({"gene": ranked_genes}).to_csv(RANKING_PATH, index=False)

	train = df[df["split"] == "train"]
	test = df[df["split"] == "test"]
	y_train = train["cancer"]
	y_test = test["cancer"]
	loo = LeaveOneOut()
	results = []

	for k in K_VALUES:
		selected_genes = ranked_genes[:k]
		pipeline = make_pipeline()
		loocv_accuracy = cross_val_score(
			pipeline,
			train[selected_genes],
			y_train,
			cv=loo,
			scoring="accuracy",
		).mean()

		pipeline.fit(train[selected_genes], y_train)
		test_accuracy = accuracy_score(y_test, pipeline.predict(test[selected_genes]))
		results.append((k, loocv_accuracy, test_accuracy))

	print("k | LOOCV accuracy | held-out test accuracy")
	print("--|----------------|-----------------------")
	for k, loocv_accuracy, test_accuracy in results:
		print(f"{k:3d} | {loocv_accuracy:14.4f} | {test_accuracy:21.4f}")

	best_k, best_loocv, best_test = max(results, key=lambda result: result[1])
	print(f"BEST_K: {best_k}")
	print(f"BEST_K LOOCV accuracy: {best_loocv:.4f}")
	print(f"BEST_K test accuracy: {best_test:.4f}")


if __name__ == "__main__":
	main()
