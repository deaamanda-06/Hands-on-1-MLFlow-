import warnings; warnings.filterwarnings("ignore")
import mlflow, optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-clf")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
Xtr_s, Xval_s = sc.transform(X_tr), sc.transform(X_val)


def train_and_eval(params):
    # slide tidak mendefinisikan train_and_eval -> dibuat dengan MLP sklearn.
    # 'dropout' dipetakan ke regularisasi L2 (alpha) karena MLP sklearn tidak punya dropout.
    m = MLPClassifier(hidden_layer_sizes=(32, 16), learning_rate_init=params["lr"],
                      batch_size=params["batch_size"], alpha=params["dropout"] * 0.1,
                      max_iter=60, random_state=42).fit(Xtr_s, y_tr)
    return f1_score(y_val, m.predict(Xval_s))


def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        val_f1 = train_and_eval(params)
        mlflow.log_metric("val_f1", val_f1)
    return val_f1


with mlflow.start_run(run_name="optuna-sweep-01"):
    mlflow.set_tag("stage", "tuning")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_f1", study.best_value)
    print("parent run:", mlflow.active_run().info.run_id)
    print("best params:", study.best_params, "best_val_f1: %.4f" % study.best_value)
