# Vineyard Lab

A small web application that predicts one of three wine cultivars from laboratory
measurements. It is an educational classification task, not a quality, price,
origin-authentication, or food-safety assessment.

The project connects a FastAPI application to a trained scikit-learn pipeline,
with DVC dataset snapshots, MLflow experiment records, container packaging and
GitHub Actions quality gates. See `docs/ACCEPTANCE.md` for verified and pending items.

## Run the application

Use Python 3.12. All commands below run from this directory.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. `/docs` provides the interactive API. A trained model
is included, so inference does not require Azure, training, or a data download.
Alternatively run `powershell -ExecutionPolicy Bypass -File scripts/start.ps1`.
If activation is blocked, invoke `.venv/Scripts/python.exe` directly.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
docker build -t vineyard:local .
docker run --rm -p 8000:8000 --env APP_ENV=local vineyard:local
```

## Training and data

Install `requirements-ml.txt`. The repository stores only `data.dvc`; the supplied
ZIP also contains a local DVC cache and an offline demonstration copy of public
data. Raw CSV files are ignored by Git. Azure is the shared remote.

The course Azure remote is now configured. Both dataset versions were uploaded
and recovered in separate clones with empty caches; see `docs/AZURE_VERIFICATION.md`.
Teammates enter the private course SAS using `python scripts/configure_course_azure.py`.
Credentials remain in ignored local configuration and are never committed.

```powershell
dvc checkout
python -m ml.train --version v2 --config ml/configs/baseline.json
python -m ml.train --version v2 --config ml/configs/tuned.json
```

Do not select models by test scores. `python -m ml.train --promote` selects from
the local experiment records by validation macro F1, checks the current dataset
hash and evaluates the selected model on the test set. Promotion writes
`models/model.joblib` and its metadata. Only promote after reviewing candidates;
do not keep retuning against the test score. See `ml/README_experiments.md`.

```powershell
python scripts/relocate_tracking.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open http://127.0.0.1:5000 and choose `vineyard-lab`. Keep `mlflow.db` and
`mlartifacts/` together. Run the relocation script after moving/extracting the
bundle, with the tracking UI stopped, then start the UI from the project directory. Tracking
is local, without a public server or credentials. MLflow is a training dependency;
the slim runtime image deliberately excludes it.

## Branches and delivery

Planned merge path: `feature/<short-description>` → `dev` → `staging` → `main`.
Feature branches start from `dev`. Every change enters protected branches by PR.

| Event | Automated behavior |
| --- | --- |
| PR to dev / staging / main | Ruff, full tests, Docker build, container smoke test |
| Push to dev | Same CI checks; no delivery |
| Push/merge to staging | CI, then delivery using the staging GitHub Environment |
| Push/merge to main | CI, then delivery using the production GitHub Environment |

Delivery checks the exact successful CI commit, writes an environment-specific
`.env` from a GitHub Environment secret, builds and runs a container on the runner,
checks environment and inference, then publishes the commit-tagged image to GHCR.
Runner hosting is temporary; no persistent public service is claimed. The PDF's
recommended staging/production image-build flow does not require persistent hosting.

**A workflow file alone cannot prevent merging.** An administrator must require
the `quality` status check on `dev`, `staging`, and `main`, require PRs, and disable
bypass where the GitHub plan supports it. See `DEPLOYMENT.md` before submission.

## Layout

`app/`: API and browser UI. `ml/`: data, training, configurations and provenance.
`models/`: small trusted runtime artifact and metadata. `tests/`: API and split
invariants. `.github/workflows/`: CI and delivery. `docs/`: acceptance checklist,
three-person preparation and bilingual viva notes.

Load only this project's trusted joblib file: Python serialization can execute code.
Future model changes should regenerate the artifact using the pinned environment.
Model limitations: 178 historical samples, no external-site validation, no confidence
calibration, and no drift monitoring. Rejecting malformed input does not establish
that valid input comes from the original data distribution.

## Sources

- Dataset and feature description: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
- Original dataset and attribution: https://archive.ics.uci.edu/dataset/109/wine
- DVC Azure configuration: https://dvc.org/doc/user-guide/data-management/remote-storage/azure-blob-storage
- MLflow tracking: https://mlflow.org/docs/latest/tracking/

The local build history records an assisted project build. It is not evidence of
three students collaborating on GitHub. Each member must review and make a real
contribution through the documented PR flow and follow the course's AI-use policy.

## Contributors

- KongYi4994 (Verified reproduction on 2026-10-18)