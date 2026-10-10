import warnings; warnings.filterwarnings("ignore")
import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
name = "breast-cancer-clf"
MARGIN = 0.0
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def pipeline_cycle(label, estimator, trusted):
    print(f"\n=== Siklus CT: {label} ===")
    # 1) Training pipeline (log ke MLflow)
    with mlflow.start_run(run_name=f"ct-{label}"):
        mlflow.set_tags({"stage": "continuous-training", "trigger": "new-data"})
        est = estimator.fit(X_tr, y_tr)
        f1_new = f1_score(y_val, est.predict(X_val))
        mlflow.log_metric("val_f1", f1_new)
        mlflow.sklearn.log_model(est, name="model", skops_trusted_types=trusted)
        run_id = mlflow.active_run().info.run_id
    # 2) Registry -> @challenger
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, "challenger", mv.version)
    # 3) Validasi otomatis vs champion pada data uji yang sama
    champ = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    f1_champ = f1_score(y_val, champ.predict(X_val))
    champ_ver = client.get_model_version_by_alias(name, "champion").version
    print(f"champion v{champ_ver} f1={f1_champ:.4f} | challenger v{mv.version} f1={f1_new:.4f} | margin={MARGIN}")
    # 4) Quality gate -> CD promote
    if f1_new >= f1_champ + MARGIN and f1_new > f1_champ:
        client.set_model_version_tag(name, mv.version, "validation_status", "passed")
        client.set_registered_model_alias(name, "champion", mv.version)
        print(f"GATE: PASSED -> v{mv.version} dipromosikan ke @champion")
    else:
        client.set_model_version_tag(name, mv.version, "validation_status", "failed")
        print(f"GATE: BLOCKED -> @champion tetap v{champ_ver}")
    print("alias sekarang: champion=v%s challenger=v%s" % (
        client.get_model_version_by_alias(name, "champion").version,
        client.get_model_version_by_alias(name, "challenger").version))


pipeline_cycle("logreg-scaled", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), None)
pipeline_cycle("tree-depth2", DecisionTreeClassifier(max_depth=2, random_state=42), ["sklearn.tree._tree.Tree"])

print()
for v in client.search_model_versions(f"name='{name}'"):
    print("v%s | aliases=%s | validation_status=%s" % (v.version, v.aliases, v.tags.get("validation_status")))
