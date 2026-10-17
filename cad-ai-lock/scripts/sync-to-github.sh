#!/usr/bin/env bash
# Sync package from Mosquito1123/zenox (branch) → Mosquito1123/cad-ai-lock
# Prerequisites (Mac): gh auth login
set -euo pipefail

DST_REPO="${DST_REPO:-Mosquito1123/cad-ai-lock}"
SRC_REPO="${SRC_REPO:-Mosquito1123/zenox}"
SRC_BRANCH="${SRC_BRANCH:-cursor/cad-ai-lock-auto-pipeline-74b7}"
PKG_PREFIX="${PKG_PREFIX:-cad-ai-lock}"
WORKDIR="${WORKDIR:-${TMPDIR:-/tmp}/cad-ai-lock-sync-$$}"

echo "==> workdir $WORKDIR"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR/src" "$WORKDIR/dst"
cd "$WORKDIR"

echo "==> fetch source package ($SRC_REPO@$SRC_BRANCH:$PKG_PREFIX)"
gh repo clone "$SRC_REPO" src -- -b "$SRC_BRANCH" --single-branch --depth 1
SRC_PKG="$WORKDIR/src/$PKG_PREFIX"
if [[ ! -d "$SRC_PKG/src" ]]; then
  echo "ERROR: expected $SRC_PKG/src" >&2
  exit 1
fi

echo "==> clone destination $DST_REPO"
gh repo clone "$DST_REPO" dst -- --depth 1
cd dst

echo "==> replace tree (keep .git)"
# wipe tracked content except .git
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$SRC_PKG"/. .

echo "==> scrub secrets / local junk"
rm -f .env .env.* !.env.example 2>/dev/null || true
rm -f .env .env.local .env.production 2>/dev/null || true
# restore example if wiped by glob
[[ -f .env.example ]] || cp "$SRC_PKG/.env.example" .env.example
rm -rf out runs .venv __pycache__ *.pem *.key 2>/dev/null || true
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

if git status --porcelain | grep -E '(^.. \.env$|\.pem$|credentials|id_rsa|id_ed25519)' ; then
  echo "ERROR: secret-looking files would be committed — abort" >&2
  exit 1
fi

chmod +x scripts/*.sh 2>/dev/null || true

git add -A
if git diff --cached --quiet; then
  echo "==> already up to date"
else
  git commit -m "Sync: privacy scrub, SECURITY.md, harden gitignore"
  git push origin HEAD:main
  echo "==> pushed"
fi

echo "DONE → https://github.com/$DST_REPO"
gh repo view "$DST_REPO"
