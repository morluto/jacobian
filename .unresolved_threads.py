import json, subprocess

prs = json.loads(subprocess.check_output(
    ["gh", "pr", "list", "--state", "open", "--limit", "200", "--json", "number,title,headRefName"]))
prs = [p for p in prs if "release" not in p["title"].lower()]

results = []
for p in prs:
    num = p["number"]
    after = ""
    threads = []
    while True:
        q = '''query($owner:String!,$repo:String!,$num:Int!,$after:String!){
          repository(owner:$owner,name:$repo){
            pullRequest(number:$num){
              reviewThreads(first:100,after:$after){
                pageInfo{hasNextPage,endCursor}
                nodes{id,isResolved,path,line,startLine,comments(first:1){nodes{author{login},body,databaseId}}}
              }
            }
          }
        }'''
        out = subprocess.run(["gh","api","graphql","-f","owner=morluto","-f","repo=jacobian",
            "-F",f"num={num}","-f",f"after={after}","-f",f"query={q}"],capture_output=True,text=True)
        if out.returncode != 0:
            print(f"ERROR {num}: {out.stderr[:300]}", flush=True)
            break
        data = json.loads(out.stdout)["data"]["repository"]["pullRequest"]["reviewThreads"]
        threads.extend(data["nodes"])
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
    unresolved = [t for t in threads if not t.get("isResolved")]
    if unresolved:
        results.append({"pr": num, "title": p["title"], "branch": p["headRefName"],
                        "count": len(unresolved),
                        "threads": [{"id": t["id"], "path": t["path"], "line": t.get("line"),
                                     "author": (t["comments"]["nodes"][0]["author"]["login"] if t["comments"]["nodes"] else None) if t["comments"]["nodes"] else None,
                                     "body": t["comments"]["nodes"][0]["body"][:500] if t["comments"]["nodes"] else ""} for t in unresolved]})
        print(f"PR {num}: {len(unresolved)} unresolved / {len(threads)} total", flush=True)

with open("/home/morluto/dev/jacobian/unresolved_report.json","w") as f:
    json.dump(results,f,indent=1)
print(f"\nTOTAL: {sum(r['count'] for r in results)} unresolved across {len(results)} PRs")
