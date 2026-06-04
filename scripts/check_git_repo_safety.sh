#!/usr/bin/env bash
set -u

failures=0

ok() {
  printf 'OK: %s\n' "$1"
}

fail() {
  printf 'FAIL: %s\n' "$1"
  failures=1
}

require_file() {
  if [ -f "$1" ]; then
    ok "$1 exists"
  else
    fail "$1 missing"
  fi
}

if [ -d .git ]; then
  ok ".git exists"
else
  fail ".git missing"
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ok "inside Git work tree"
else
  fail "not inside Git work tree"
fi

require_file ".gitignore"
require_file "README.md"
require_file "ARTIFACTS.md"
require_file "ENVIRONMENT.md"
require_file "GIT_INITIALIZATION_REPORT.md"

tracked_tmp="/tmp/sc_explorer_safety_tracked_files.txt"
forbidden_tmp="/tmp/sc_explorer_safety_forbidden_tracked.txt"
large_tmp="/tmp/sc_explorer_safety_tracked_large_files.txt"
status_tmp="/tmp/sc_explorer_safety_status.txt"
unexpected_tmp="/tmp/sc_explorer_safety_unexpected_status.txt"

git ls-files > "$tracked_tmp"

if grep -E '^(outputs/|logs/|checkpoints/|data/|assets/home_like_scene_v1/current_environment/dependencies/|assets/home_like_scene_v1/current_environment_localized/|assets/home_like_scene_v1/current_environment_localized_defaultprim/|building_scene\.usd$)' "$tracked_tmp" > "$forbidden_tmp"; then
  fail "forbidden artifact paths are tracked"
  sed -n '1,80p' "$forbidden_tmp"
else
  ok "no tracked forbidden artifact paths"
fi

: > "$large_tmp"
while IFS= read -r -d '' f; do
  if [ -f "$f" ]; then
    size=$(stat -c%s "$f")
    if [ "$size" -gt 52428800 ]; then
      printf '%s %s\n' "$f" "$size" >> "$large_tmp"
    fi
  fi
done < <(git ls-files -z)

if [ -s "$large_tmp" ]; then
  fail "tracked files larger than 50 MB found"
  sed -n '1,80p' "$large_tmp"
else
  ok "no tracked files larger than 50 MB"
fi

git status --short --untracked-files=all > "$status_tmp"
grep -vE '^\?\? git_initial_commit_hash\.txt$' "$status_tmp" > "$unexpected_tmp" || true

if [ -s "$unexpected_tmp" ]; then
  fail "working tree has unexpected status entries"
  sed -n '1,80p' "$unexpected_tmp"
else
  ok "working tree clean or only expected git_initial_commit_hash.txt is untracked"
fi

if [ "$failures" -eq 0 ]; then
  printf 'RESULT: PASS\n'
  exit 0
fi

printf 'RESULT: FAIL\n'
exit 1
