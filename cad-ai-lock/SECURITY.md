# Security notes

## Secrets

- Put API keys only in a local `.env` (see `.env.example`).
- `.env` is gitignored. Do **not** commit tokens, cookies, or private keys.
- Env vars used by this project: `REPLICATE_API_TOKEN`, `FAL_KEY`, `OPENAI_API_KEY`.

## Paths

- Extracted `RoomPlan.source_path` is stored as a **basename only** (no home/absolute paths), so run artifacts are safer to share.

## Reporting

If you find a credential leak in history, rotate the key immediately and open an issue (without pasting the secret).
