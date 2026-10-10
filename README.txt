Bukti uji coba Hands-on 1 (MLflow) - Modul Model Experimentation & Tracking, Minggu 6
Lingkungan: Python 3.12.3, MLflow 3.17.0, scikit-learn 1.8.0, Optuna 5.0.0

Urutan menjalankan ulang:
 1. mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
 2. python skrip/t1_basic.py            # logging dasar (slide 42-43)
 3. python skrip/t2_epoch_autolog.py    # logging per epoch + autolog (slide 44-46)
 4. python skrip/t3_optuna.py           # nested runs Optuna (slide 47-48)
 5. python skrip/t4_search.py           # search_runs (slide 49)
 6. python skrip/t5_registry.py         # registry + alias (slide 50); ganti best_run_id dengan run rf-baseline Anda
 7. MLFLOW_TRACKING_URI=http://localhost:5000 mlflow models serve -m "models:/breast-cancer-clf@champion" -p 8080 --env-manager local
 8. python skrip/t6_cicd.py             # simulasi quality gate CI/CD/CT (slide 51)
Catatan: run_id di t5_registry.py dan shots2.py adalah milik sesi uji ini; ganti dengan run_id Anda.
