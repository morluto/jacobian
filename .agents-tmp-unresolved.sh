#!/usr/bin/env bash
# Fetch unresolved review threads for all open PRs except the release PR.
set -euo pipefail
cd /home/morluto/dev/jacobian

# Get all open PR numbers except release PR
mapfile -t PRS < <(gh pr list --state open --json number,title --limit 300 \
  --jq '.[] | select(.title | test("release";"i") | not) | .number')

echo "Checking ${#PRS[@]} PRs for unresolved threads..."

for pr in "${PRS[@]}"; do
  out=$(gh api graphql -f query='
    query($owner:String!, $repo:String!, $pr:Int!) {
      repository(owner:$owner, name:$repo) {
        pullRequest(number:$pr) {
          reviewThreads(first:100) {
            nodes {
              isResolved isOutdated
              comments(first:1) { nodes { author{login} body databaseId path } }
            }
          }
        }
      }
    }' -F owner=morluto -F repo=jacobian -F pr="$pr" --jq '
    [.data.repository.pullRequest.reviewThreads.nodes[]
     | select(.isResolved == false)
     | {resolved: .isResolved, outdated: .isOutdated,
        path: .comments.nodes[0].path,
        id: .comments.nodes[0].databaseId,
        author: (.comments.nodes[0].author.login // "ghost"),
        body: .comments.nodes[0].body}]
    | if length > 0 then {pr: '"$pr"', threads: .} else empty end')
  if [ -n "$out" ]; then
    echo "$out"
  fi
done
