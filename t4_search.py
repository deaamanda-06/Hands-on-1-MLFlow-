import warnings; warnings.filterwarnings("ignore")
import mlflow, pandas as pd
pd.set_option("display.width", 200)
mlflow.set_tracking_uri("http://localhost:5000")
all_runs = mlflow.search_runs(experiment_names=["breast-cancer-clf"], output_format="pandas")
print("total runs:", len(all_runs))
parent = all_runs[all_runs["tags.mlflow.runName"] == "optuna-sweep-01"].iloc[0]["run_id"]
children = all_runs[all_runs["tags.mlflow.parentRunId"] == parent]
print("nested child runs under parent:", len(children))
print("best child val_f1: %.4f, worst: %.4f" % (children["metrics.val_f1"].max(), children["metrics.val_f1"].min()))

# --- persis seperti slide 49 ---
runs = mlflow.search_runs(
    experiment_names=["breast-cancer-clf"],
    filter_string="metrics.val_f1 > 0.95 and params.max_depth = '8' and tags.stage != 'debug'",
    order_by=["metrics.val_f1 DESC"],
    max_results=10,
)
print("\nSLIDE 49 query ->", len(runs), "run")
print(runs[["run_id", "params.n_estimators", "metrics.val_f1", "metrics.val_auc"]])
best_run_id = runs.iloc[0]["run_id"]
print("best_run_id:", best_run_id)

# --- query tambahan: seluruh run dengan val_f1 > 0.95 ---
r2 = mlflow.search_runs(experiment_names=["breast-cancer-clf"], filter_string="metrics.val_f1 > 0.95",
                        order_by=["metrics.val_f1 DESC"], max_results=5)
print("\nTop run seluruh experiment (val_f1 > 0.95):")
print(r2[["tags.mlflow.runName", "metrics.val_f1", "metrics.val_auc"]])
