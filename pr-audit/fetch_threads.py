import json, subprocess

prs = json.loads(subprocess.check_output(
    ["gh", "pr", "list", "--state", "open", "--limit", "200", "--json", "number,title,headRefName"]))
prs = [p for p in prs if p["number"] != 2104]

query = """
query($o: String!, $n: String!, $num: Int!) {
  repository(owner: $o, name: $n) {
    pullRequest(number: $num) {
      number
      title
      headRefName
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 20) {
            nodes {
              author { login }
              body
              createdAt
              replyTo { id }
            }
          }
        }
      }
    }
  }
}
"""

out = {}
for p in prs:
    r = json.loads(subprocess.check_output(
        ["gh", "api", "graphql", "-f", f"query={query}", "-f", "o=morluto", "-f", "n=jacobian",
         "-F", f"num={p['number']}"]))
    data = r["data"]["repository"]["pullRequest"]
    threads = [t for t in data["reviewThreads"]["nodes"] if not t["isResolved"]]
    if threads:
        out[p["number"]] = {"title": p["title"], "headRefName": p["headRefName"], "threads": threads}

with open("/home/morluto/dev/jacobian/pr-audit/unresolved.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"{len(out)} PRs with unresolved threads")
for num in sorted(out):
    t = out[num]["threads"]
    print(num, len(t), out[num]["title"][:60])
