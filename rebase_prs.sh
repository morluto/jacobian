#!/usr/bin/env bash
set -eu

export PATH="$HOME/.local/bin:$PATH"
cd /home/morluto/dev/jacobian

BRANCHES=(
  agent/harbor-continuous-spike-integral-separation
  agent/harbor-pythagorean-generator-recurrence
  agent/harbor-lp-integrability-separator
  agent/harbor-even-fixed-point-inclusion
  agent/harbor-grid-independent-set-transfer
  agent/harbor-necklace-burnside-certificate
  agent/harbor-image-complement-commutation
  agent/harbor-radical-distance-triangle-certificate
  agent/harbor-permutation-inversion-involution
  agent/harbor-distinct-parity-primal-dual
  agent/harbor-unit-fraction-classification-repair
  agent/harbor-chebotarev-proof-audit
  agent/harbor-inversion-aggregate-mask
  agent/harbor-alternating-recurrence-stability-certificate
  agent/harbor-rank-one-spectral-limit-certificate
  agent/harbor-primitive-eisenstein-norm-audit
  agent/harbor-finite-support-sum-scope-audit
  agent/harbor-divisor-sum-square-sequence-repair
  agent/harbor-gf2-matrix-completion-quantifier-audit
  agent/harbor-valuation-gcd-quantifier-audit
  agent/harbor-quartic-chebotarev-proof-repair
  agent/harbor-cyclic-lipschitz-duality
  agent/harbor-edge-pair-ordering-audit
  agent/harbor-closed-one-form-classification
  agent/harbor-finite-field-irreducibility-repair
  agent/harbor-bounded-variation-uniform-limit
  agent/harbor-path-dependent-limit
  agent/harbor-metamath-syllogism-repair
  agent/harbor-multiplicative-grid-extremum
)

FAILED=()
SUCCESS=()

for branch in "${BRANCHES[@]}"; do
  echo "=== Processing $branch ==="

  # Clean working tree
  git checkout main 2>/dev/null
  git clean -fdx -- benchmarks/ .jacobian/ 2>/dev/null || true

  # Checkout the branch
  if ! git checkout "$branch" 2>/dev/null; then
    echo "  SKIP: cannot checkout $branch"
    FAILED+=("$branch (checkout)")
    continue
  fi

  # Rebase onto main
  if ! git rebase origin/main 2>/dev/null; then
    echo "  CONFLICT: rebase failed for $branch"
    git rebase --abort 2>/dev/null || true
    FAILED+=("$branch (rebase conflict)")
    git checkout main 2>/dev/null
    continue
  fi

  # Run harbor-sync to fix checksums + formatting
  make harbor-sync 2>/dev/null || true

  # Check if there are changes to commit
  if ! git diff --quiet; then
    git add -A
    git commit --amend --no-edit 2>/dev/null
    echo "  amended commit with harbor-sync fixes"
  else
    echo "  no harbor-sync changes needed"
  fi

  # Force-push
  if git push --force-with-lease origin "$branch" 2>/dev/null; then
    echo "  PUSHED"
    SUCCESS+=("$branch")
  else
    echo "  PUSH FAILED"
    FAILED+=("$branch (push)")
  fi

  git checkout main 2>/dev/null
  echo
done

echo "=== SUMMARY ==="
echo "Success: ${#SUCCESS[@]}"
for s in "${SUCCESS[@]}"; do echo "  $s"; done
echo "Failed: ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  $f"; done
