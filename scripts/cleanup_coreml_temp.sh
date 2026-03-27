#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
TMPROOT="${TMPDIR%/}"
if [[ -z "${TMPROOT}" ]]; then
  TMPROOT="/private/var/folders"
fi

TARGETS=(
  "${TMPROOT}/cohere_*.mlmodelc"
  "${TMPROOT}/tmp*.mlpackage"
  "${TMPROOT}/pure_cli_*"
)

echo "TMPROOT=${TMPROOT}"
echo "DRY_RUN=${DRY_RUN}"

for pattern in "${TARGETS[@]}"; do
  shopt -s nullglob
  matches=( $pattern )
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    continue
  fi
  echo "matched pattern: ${pattern}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  %s\n' "${matches[@]}"
  else
    rm -rf "${matches[@]}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry-run complete"
else
  echo "cleanup complete"
fi
