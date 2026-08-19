# Attrition Intelligence

Attrition Intelligence is a full-stack employee attrition risk assessment application for HR teams. It combines a React dashboard, a Node.js/tRPC server, authenticated prediction history, and a Python inference runner that loads the validated `best_f1_model.joblib` artifact.

> The application is a decision-support tool. It does not make automated employment decisions. Predictions should be reviewed alongside employee context, appropriate governance, and human judgment.

## Product capabilities

The dashboard presents the validated model metrics **CV F1: 0.6536**, **Test F1: 0.5882**, and **PR-AUC: 0.6086**. Authenticated users can submit employee attributes, receive a real-time attrition probability, choose an operating mode, and review saved assessments.

| Operating mode | Threshold | Intended use |
|---|---:|---|
| High Recall | 0.27 | Broader screening coverage |
| Balanced | 0.32 | Recommended default operating point |
| High Precision | 0.45 | Fewer false alerts |

The Model Insights view is backed by experiment artifacts rather than placeholder values. It includes the precision/recall frontier, measured feature-importance values, and constrained operating-point information.

## Architecture

The application is organized around four runtime layers:

| Layer | Implementation | Responsibility |
|---|---|---|
| Frontend | React 19, Vite, Tailwind CSS, shadcn/ui | Dashboard, prediction form, history, and model insights |
| Application server | Express, tRPC, TypeScript | Authentication-aware procedures and API contracts |
| ML runtime | Python 3, scikit-learn, XGBoost, joblib | Load `best_f1_model.joblib` and calculate probabilities |
| Persistence | MySQL/TiDB through Drizzle ORM | Authenticated prediction history and user records |

The Node.js server invokes `server/ml/infer.py` as a child process and sends a JSON employee payload through standard input. The Python process returns a JSON response containing the probability, model artifact name, and all three operating-mode decisions.

Production inference is intentionally decoupled from the training-only LightGBM and CatBoost imports. This prevents optional native training libraries from blocking model loading. The production Python manifest contains only the dependencies required by the serialized inference artifact.

## Repository structure

```text
client/                         React application and dashboard pages
drizzle/                        Database schema and migrations
ml/                             Repository ML workflows and experiment code
ml_artifacts/                   Validated model and experiment artifacts
server/                         tRPC procedures, database helpers, and ML bridge
server/ml/best_f1_model.joblib  Runtime model artifact
server/ml/infer.py              Python inference entry point
shared/                         Shared TypeScript constants and types
Dockerfile                      Node.js and Python production runtime
requirements-python.txt         Python inference dependencies
package.json                    Node.js scripts and dependencies
```

## Prerequisites

For local development, install Node.js 22 or later, pnpm, Python 3.11 or later, and a MySQL-compatible database. The application expects the standard Manus environment variables for authentication and database access when run inside the managed environment.

The Python inference runtime requires the packages listed in `requirements-python.txt`. Install them with:

```bash
python3 -m pip install -r requirements-python.txt
```

Install JavaScript dependencies with:

```bash
pnpm install
```

## Configuration

Do not commit `.env` files or credentials. Configure the following values through the deployment environment or local secret manager:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | MySQL/TiDB connection string |
| `JWT_SECRET` | Session-cookie signing secret |
| `VITE_APP_ID` | OAuth application identifier |
| `OAUTH_SERVER_URL` | OAuth server base URL |
| `VITE_OAUTH_PORTAL_URL` | Frontend login portal URL |

The complete set of managed environment variables is defined by the deployment template. The application reads server-side configuration through `server/_core/env.ts`.

## Running locally

Start the development server with:

```bash
pnpm dev
```

Run TypeScript validation with:

```bash
pnpm check
```

Run the complete Vitest suite with:

```bash
pnpm test -- --run
```

Create a production build with:

```bash
pnpm build
```

The repository includes a real authenticated smoke-test integration that loads the model artifact and exercises the protected prediction creation path. The persistence test is cleanup-safe and verifies the saved probability, mode, threshold, and serialized inputs against the real database when `DATABASE_URL` is available.

## Database migrations

The prediction history table is defined in `drizzle/schema.ts`. The migration that widened the probability column to support full-precision model output is stored in `drizzle/0002_futuristic_marvel_zombies.sql`.

For schema changes, generate the migration first, review the SQL, and apply it through the managed database workflow:

```bash
pnpm drizzle-kit generate
```

The `predictions.probability` column is `varchar(32)` because model outputs such as `0.018643820763448102` exceed the previous 16-character limit.

## ML inference contract

The inference runner accepts a JSON employee payload containing the HR features used by the validated model, including age, department, job role, overtime, satisfaction indicators, compensation, tenure, travel, and work-life variables.

Example invocation:

```bash
python3 server/ml/infer.py < server/ml/sample_input.json
```

The response has the following shape:

```json
{
  "probability": 0.018643820763448102,
  "modelArtifact": "best_f1_model.joblib",
  "modes": {
    "high_recall": { "threshold": 0.27, "predictedAttrition": false },
    "balanced": { "threshold": 0.32, "predictedAttrition": false },
    "high_precision": { "threshold": 0.45, "predictedAttrition": false }
  }
}
```

## Model limitations

The model is based on a historical employee-attrition dataset and should not be interpreted as evidence of causal risk. The dataset does not establish a longitudinal prediction horizon, feature-availability timestamp, external validation population, or production drift baseline. Its measured test F1 is **0.5882**, and no leakage-free experiment achieved the requested F1 target of 0.85.

HR teams should avoid using protected attributes or model outputs as the sole basis for employment decisions. The application should be used to support structured conversations and investigate patterns, not to automate adverse action.

## Production deployment

The included `Dockerfile` provides both Node.js and Python runtimes. It installs the OpenMP runtime required by native numerical libraries and runs the managed Node.js application process. The application is compatible with the managed deployment workflow used by the project.

Before deployment, verify:

1. `best_f1_model.joblib` is present in the runtime image.
2. `requirements-python.txt` is installed successfully.
3. `DATABASE_URL` points to the intended database.
4. `pnpm check`, `pnpm test -- --run`, and `pnpm build` pass.
5. An authenticated prediction can be created and appears in History.

## Troubleshooting

If inference reports a Python traceback, run the sample payload locally and inspect the returned JSON. If model loading fails, confirm that the artifact path and Python dependencies are present in the image. If prediction persistence fails with a data-length error, confirm that the database has applied the migration widening `predictions.probability` to `varchar(32)`.

If a user cannot access predictions or history, confirm that the request is authenticated and that the tRPC context contains the expected user. The prediction procedures are protected and intentionally reject unauthenticated requests.

## License

No license has been selected for this repository yet. Add an explicit license before distributing the project outside the current team.
