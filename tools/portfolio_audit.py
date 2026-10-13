"""Read-only portfolio inventory for Physica donor discovery.

Uses the authenticated GitHub CLI ONLY to obtain a short-lived API token.
Writes the report solely to the requested path; do not commit private scans.
No code is copied or imported from donor repositories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen


def _token():
    p = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    return p.stdout.strip()


def _get(url, token):
    request = Request(url, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "User-Agent": "physica-read-only-portfolio-audit",
    })
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def scan(owner, out):
    token = _token()
    repos = []
    for page in range(1, 11):
        batch = _get(
            f"https://api.github.com/user/repos?affiliation=owner&per_page=100&page={page}",
            token,
        )
        repos.extend(r for r in batch if r["owner"]["login"].lower() == owner.lower())
        if len(batch) < 100:
            break
    results = []
    needles = ("physics", "simul", "model", "ode", "pde", "engine",
               "experiment", "invariant", "validation", "provenance", "numeri",
               "control", "benchmark", "dynamics", "integrat", "reproduc")

    def examine(r):
        name = r["full_name"]
        entry = {
            "repo": name, "private": r["private"], "archived": r["archived"],
            "default_branch": r["default_branch"], "tree_checked": False,
            "tree_truncated": None, "tree_sha": None, "file_count": None,
            "candidate_paths": [], "error": None,
        }
        try:
            branch = quote(r["default_branch"], safe="")
            tree = _get(
                f"https://api.github.com/repos/{name}/git/trees/{branch}?recursive=1",
                token,
            )
            paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"]
            entry.update(
                tree_checked=True, tree_truncated=tree.get("truncated", False),
                tree_sha=tree["sha"], file_count=len(paths),
                candidate_paths=[p for p in paths if any(x in p.lower() for x in needles)][:40],
            )
        except Exception as exc:
            entry["error"] = type(exc).__name__ + ": " + str(exc)[:160]
        return entry

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(examine, repo): repo for repo in repos}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["repo"].casefold())
    report = {
        "scope": "authenticated owner portfolio",
        "owner": owner,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "repository_count": len(results),
        "tree_reads_succeeded": sum(r["tree_checked"] for r in results),
        "tree_reads_failed": sum(not r["tree_checked"] for r in results),
        "private_repo_count": sum(r["private"] for r in results),
        "results": results,
        "authority": "read-only structural inventory; no source-code reuse asserted",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="thebrazenbeard")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scan(args.owner, args.out)
