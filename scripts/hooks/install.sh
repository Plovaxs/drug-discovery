#!/usr/bin/env bash
# Installs this repo's git hooks. Hooks are not tracked by git, so the tracked copies live in
# scripts/hooks/ and this script links them into .git/hooks/ -- editing the tracked copy then takes
# effect immediately, and a fresh clone needs one command rather than a remembered manual step.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_SRC="$REPO_ROOT/scripts/hooks"
HOOK_DST="$(git rev-parse --git-path hooks)"

mkdir -p "$HOOK_DST"
for hook in pre-commit; do
  src="$HOOK_SRC/$hook"
  dst="$HOOK_DST/$hook"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    # Never clobber a hook someone wrote by hand without leaving them a copy.
    mv "$dst" "$dst.replaced-$(date +%Y%m%d%H%M%S)"
    echo "existing $hook moved aside"
  fi
  ln -sf "$src" "$dst"
  chmod +x "$src"
  echo "installed $hook -> $src"
done
echo "done. bypass a single commit with: git commit --no-verify"
