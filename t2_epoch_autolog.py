import mlflow, hashlib, numpy as np, pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-clf")
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
Xtr_s, Xval_s = sc.transform(X_tr), sc.transform(X_val)
dataset_md5 = hashlib.md5(X.tobytes()).hexdigest()
open("requirements.txt", "w").write("scikit-learn==1.8.0\nmlflow==3.17.0\n")

# --- (A) per-epoch logging (analog slide 44-45; MLP sklearn sebagai pengganti PyTorch) ---
cfg = {"lr": 3e-3, "batch_size": 64, "epochs": 20, "arch": "mlp-32-16", "seed": 42}
np.random.seed(cfg["seed"])
net = MLPClassifier(hidden_layer_sizes=(32, 16), learning_rate_init=cfg["lr"], random_state=cfg["seed"])
classes = np.unique(y_tr)
with mlflow.start_run(run_name=f"{cfg['arch']}-lr{cfg['lr']}"):
    mlflow.set_tag("stage", "dl-pattern")
    mlflow.log_params(cfg)
    mlflow.log_param("data_md5", dataset_md5)
    mlflow.log_artifact("requirements.txt")
    best_f1 = 0.0
    for epoch in range(cfg["epochs"]):
        idx = np.random.permutation(len(Xtr_s))
        for i in range(0, len(idx), cfg["batch_size"]):
            b = idx[i:i + cfg["batch_size"]]
            net.partial_fit(Xtr_s[b], y_tr[b], classes=classes)
        train_loss = log_loss(y_tr, net.predict_proba(Xtr_s))
        val_loss = log_loss(y_val, net.predict_proba(Xval_s))
        val_f1 = f1_score(y_val, net.predict(Xval_s))
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                            "val_f1": val_f1, "lr": cfg["lr"]}, step=epoch)
        if val_f1 > best_f1:
            best_f1 = val_f1
            pickle.dump(net, open("best.pkl", "wb"))
            mlflow.log_artifact("best.pkl", artifact_path="checkpoints")
    mlflow.log_metric("best_val_f1", best_f1)
    mlflow.sklearn.log_model(net, name="model", skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"])
    print("A) per-epoch run:", mlflow.active_run().info.run_id, "best_val_f1=%.4f" % best_f1)

# --- (B) autolog (slide 46) ---
mlflow.sklearn.autolog()
with mlflow.start_run(run_name="rf-autolog"):
    m = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(X_tr, y_tr)
    mlflow.log_param("data_md5", dataset_md5)   # praktik terbaik: log manual versi data
    print("B) autolog run:", mlflow.active_run().info.run_id)
