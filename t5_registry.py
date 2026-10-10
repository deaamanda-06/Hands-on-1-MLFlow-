import warnings; warnings.filterwarnings("ignore")
import mlflow, numpy as np
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
best_run_id = "c77ac8f535e9442691a7365d643275cd"
client = MlflowClient()
name = "breast-cancer-clf"

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Registrasi
mv = mlflow.register_model(f"runs:/{best_run_id}/model", name)
print("1. registered:", mv.name, "version", mv.version)
# 2. Anotasi & challenger
client.update_model_version(name, mv.version, description="RF depth=8, data v3")
client.set_model_version_tag(name, mv.version, "validation_status", "passed")
client.set_registered_model_alias(name, "challenger", mv.version)
print("2. alias challenger ->", client.get_model_version_by_alias(name, "challenger").version)
# 3. Promote
client.set_registered_model_alias(name, "champion", mv.version)
print("3. alias champion ->", client.get_model_version_by_alias(name, "champion").version)
# 4. Load via alias
model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
preds = model.predict(X_val)
print("4. loaded via @champion; acc on val = %.4f" % (np.asarray(preds) == y_val).mean(), "| n preds:", len(preds))
mvv = client.get_model_version(name, mv.version)
print("   description:", mvv.description, "| tags:", dict(mvv.tags), "| aliases:", mvv.aliases)
