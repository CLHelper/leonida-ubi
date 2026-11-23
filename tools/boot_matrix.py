"""How far the retail executable boots for each named set of pull requests: one row per set.

Boot work on RT-5 is spread over many open pull requests, and the stop a contributor reports depends on which of
them were merged, in which order, onto which `main`, and on whether the shader library matched the translator. This
tool makes that explicit in three steps:

    python -I tools/boot_matrix.py merge --base origin/main --set rt5=7286,8517 --set rt5-6891=7286,8517,6891 \\
        --fetch origin --out <matrix folder>
    python -I tools/boot_matrix.py run <matrix folder> --set rt5 --tree <that set's checkout or export> \\
        --exe <game_scarlett_master.exe> --game-dir <package> --shaders <work>/shaders/spirv --build
    python -I tools/boot_matrix.py report <matrix folder>

`merge` (git only; nothing from a pull request is executed). For each set, in the order given, every PR is merged onto
the result so far with `git merge-tree --write-tree` and `git commit-tree`: no checkout, no hooks, the working tree is
not touched. A PR that `scripts/merge_check.py` finds already on the base (`already-merged`, the case where a stop
was fixed by a merge the stack predates) is skipped, and one that conflicts with the PRs before it is skipped with
its conflicted files listed. The result is the branch `boot-matrix/<set>` and `<out>/matrix.json`. A PR spec is `N`
(the ref `refs/boot-matrix/pr/N`, fetched from `pull/N/head` with `--fetch REMOTE`) or `N:REF` for a ref you have.
The merge commits use a fixed author and the base's commit date, so the same inputs give the same commit.

`run` builds and boots one set's tree: export or check out `boot-matrix/<set>` where you build PR code (another
machine, a separate user), and point `--tree` at it. With `--build` it configures and builds `runtime/` there; else
`--gta6rt` names the executable. It runs `gta6rt --run --trace --trace-d3d12` with `LEONIDA_D3D12=vulkan` and
`LEONIDA_SHADERS`, stops the process tree at `--timeout`, and writes `<out>/<set>/runN/result.json`: the stop, the
D3D12 call count and last call (summarised with `tools/boot_trace.py`), and whether the shader library was
translated by this tree's translator (`translation_sources.sha256` that `make_game.py` writes beside it, against
the same digest over the tree's `TRANSLATION_SOURCES`). A library from another translator is the usual reason a
reported stop does not reproduce.

`report` prints a Markdown table (or `--json`) with the newest run of each set, ready to paste into an issue.

What a row says: `exit N` (the entry point returned), `running at timeout` (still going when stopped), `last
exception CODE at SITE` (the process ended without an exit event; exceptions are recorded before handlers run, so
the last one is where to look first, not proof of the cause), `error` (the loader threw) or `no trace`. Rows carry
counts, RVAs, method and import names only; the traces stay in the matrix folder on your machine (AGENTS.md, "No
game data"), and failed file paths and debug strings are left out of the report.

Standard library only; needs git 2.38 or newer for `merge-tree --write-tree`. Exit 0 on success, 1 when a set had a
conflict or a run did not reach a trace, 2 on a usage, git or file error.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import boot_trace  # noqa: E402
import merge_check  # noqa: E402

MATRIX = "matrix.json"
RESULT = "result.json"
PR_REF = "refs/boot-matrix/pr/{}"
BRANCH = "boot-matrix/{}"
SET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MERGE_AUTHOR = {"GIT_AUTHOR_NAME": "boot_matrix", "GIT_AUTHOR_EMAIL": "boot-matrix@invalid",
                "GIT_COMMITTER_NAME": "boot_matrix", "GIT_COMMITTER_EMAIL": "boot-matrix@invalid"}
DEFAULT_TIMEOUT = 600   # seconds; the retail boot reaches its frame loop well within this on a desktop GPU
GTA6RT_PATHS = ("gta6rt.exe", os.path.join("RelWithDebInfo", "gta6rt.exe"), os.path.join("Release", "gta6rt.exe"),
                os.path.join("Debug", "gta6rt.exe"))
TAIL_LINES = 20


class MatrixError(Exception):
    """A usage, git or file problem that stops the step (exit 2)."""


# --- set specs and git ---------------------------------------------------------------------------------------------

def parse_set(text: str) -> tuple[str, list[dict]]:
    """`name=7286,8517:refs/x` -> ("name", [{"pr": 7286, "ref": "refs/boot-matrix/pr/7286"}, ...]).

    `name=` is the base alone, the row the other sets are compared with."""
    name, sep, specs = text.partition("=")
    if not sep or not SET_NAME.match(name):
        raise argparse.ArgumentTypeError(f"expected NAME=PR[,PR...], got {text!r}")
    if not specs:
        return name, []
    prs = []
    for spec in specs.split(","):
        number, _, ref = spec.strip().partition(":")
        number = number.lstrip("#")
        if not number.isdigit():
            raise argparse.ArgumentTypeError(f"{spec!r} in set {name!r}: expected N or N:REF")
        prs.append({"pr": int(number), "ref": ref or PR_REF.format(int(number))})
    if len({p["pr"] for p in prs}) != len(prs):
        raise argparse.ArgumentTypeError(f"set {name!r} names a PR twice")
    return name, prs


def git(repo: str, *args: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    full_env = None if env is None else {**os.environ, **env}
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, env=full_env)
    r.stdout = r.stdout.decode("utf-8", "replace")
    r.stderr = r.stderr.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise MatrixError(f"git {' '.join(args)}: {r.stderr.strip() or r.returncode}")
    return r


def rev(repo: str, ref: str) -> str:
    r = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    if r.returncode != 0:
        raise MatrixError(f"no commit for {ref!r} (fetch it, or give --fetch REMOTE)")
    return r.stdout.strip()


def fetch(repo: str, remote: str, numbers: list[int]) -> None:
    refspecs = [f"+pull/{n}/head:{PR_REF.format(n)}" for n in numbers]
    git(repo, "fetch", "--quiet", "--no-tags", remote, *refspecs)


# --- the translator digest make_game.py keeps beside the shader library ------------------------------------------

def translation_sources(make_game_text: str) -> tuple[str, ...]:
    """The TRANSLATION_SOURCES tuple of a make_game.py, read without running it."""
    for node in ast.parse(make_game_text).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRANSLATION_SOURCES" for t in node.targets):
            names = ast.literal_eval(node.value)
            if isinstance(names, tuple) and all(isinstance(n, str) for n in names):
                return names
    raise MatrixError("make_game.py has no TRANSLATION_SOURCES tuple of file names")


def translation_digest(read) -> str:
    """make_game.translation_sources_digest() over files given by read(relative path) -> bytes."""
    names = translation_sources(read("make_game.py").decode("utf-8"))
    h = hashlib.sha256(b"")
    for name in names:
        data = read(f"tools/{name}")
        h.update(f"{name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def digest_of_commit(repo: str, commit: str) -> str:
    def read(path: str) -> bytes:
        r = subprocess.run(["git", "-C", repo, "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
        if r.returncode != 0:
            raise MatrixError(f"{commit[:12]} has no {path}")
        return r.stdout
    return translation_digest(read)


def digest_of_tree(tree: str) -> str:
    def read(path: str) -> bytes:
        try:
            with open(os.path.join(tree, *path.split("/")), "rb") as f:
                return f.read()
        except OSError as e:
            raise MatrixError(f"cannot read {path} in {tree}: {e.strerror or e}") from e
    return translation_digest(read)


def shader_library_state(shaders: str, tree_digest: str) -> str:
    try:
        with open(os.path.join(shaders, "translation_sources.sha256"), encoding="utf-8") as f:
            recorded = f.read().strip()
    except OSError:
        return "unknown"
    return "current" if recorded == tree_digest else "stale"


# --- merge ---------------------------------------------------------------------------------------------------------

def merge_set(repo: str, base: str, name: str, prs: list[dict], checks: dict, env: dict) -> dict:
    head = base
    rows = []
    for pr in prs:
        row = {"pr": pr["pr"], "ref": pr["ref"], "sha": pr["sha"], "verdict": checks[pr["pr"]]["verdict"],
               "duplicates": sum(checks[pr["pr"]]["duplicates"].values()),
               "conflicts_with_base": checks[pr["pr"]]["conflict"], "conflict_files": []}
        if row["verdict"] == "already-merged":
            row["status"] = "already-merged"
            rows.append(row)
            continue
        r = git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", head, pr["sha"], check=False)
        lines = [line for line in r.stdout.splitlines() if line]
        if r.returncode == 1:
            row["status"] = "conflict"
            row["conflict_files"] = lines[1:]
        elif r.returncode != 0 or not lines:
            raise MatrixError(f"git merge-tree for #{pr['pr']} in set {name}: {r.stderr.strip() or r.returncode}")
        else:
            message = f"boot_matrix: merge #{pr['pr']} into set {name}"
            head = git(repo, "commit-tree", lines[0], "-p", head, "-p", pr["sha"], "-m", message,
                       env=env).stdout.strip()
            row["status"] = "merged"
        rows.append(row)
    branch = BRANCH.format(name)
    git(repo, "update-ref", f"refs/heads/{branch}", head)
    return {"name": name, "branch": branch, "head": head, "prs": rows,
            "translation_digest": digest_of_commit(repo, head)}


def cmd_merge(a: argparse.Namespace) -> int:
    repo = os.path.abspath(a.repo)
    sets = dict(a.set)
    if len(sets) != len(a.set):
        raise MatrixError("two sets share a name")
    numbers = sorted({p["pr"] for prs in sets.values() for p in prs if p["ref"] == PR_REF.format(p["pr"])})
    if a.fetch and numbers:
        fetch(repo, a.fetch, numbers)
    base = rev(repo, a.base)
    for prs in sets.values():
        for pr in prs:
            pr["sha"] = rev(repo, pr["ref"])
    unique = {pr["pr"]: pr["sha"] for prs in sets.values() for pr in prs}
    try:
        history = merge_check.prior_merges(repo, base, unique, a.depth)
        checks = {n: merge_check.check(repo, base, n, sha, a.min_length, history[n]) for n, sha in unique.items()}
    except merge_check.GitError as e:
        raise MatrixError(str(e)) from e
    date = git(repo, "show", "-s", "--format=%cI", base).stdout.strip()
    env = {**MERGE_AUTHOR, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    matrix = {"tool": "boot_matrix", "version": 1, "base": {"ref": a.base, "sha": base},
              "sets": [merge_set(repo, base, name, prs, checks, env) for name, prs in sets.items()]}
    os.makedirs(a.out, exist_ok=True)
    write_json(os.path.join(a.out, MATRIX), matrix)
    for s in matrix["sets"]:
        merged = sum(1 for p in s["prs"] if p["status"] == "merged")
        skipped = [f"#{p['pr']} {skip_reason(p)}" for p in s["prs"] if p["status"] != "merged"]
        print(f"{s['name']}: {s['branch']} {s['head'][:12]}, {merged}/{len(s['prs'])} merged"
              + (f"; skipped {', '.join(skipped)}" if skipped else ""))
    conflicts = any(p["status"] == "conflict" for s in matrix["sets"] for p in s["prs"])
    return 1 if conflicts else 0


# --- run -----------------------------------------------------------------------------------------------------------

def read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise MatrixError(f"cannot read {path}: {e}") from e


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")


def find_set(matrix: dict, name: str) -> dict:
    for s in matrix["sets"]:
        if s["name"] == name:
            return s
    raise MatrixError(f"no set {name!r} in the matrix ({', '.join(s['name'] for s in matrix['sets'])})")


def next_run_dir(set_dir: str) -> str:
    os.makedirs(set_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(set_dir, f"run{n}")):
        n += 1
    path = os.path.join(set_dir, f"run{n}")
    os.makedirs(path)
    return path


def tail(path: str, lines: int = TAIL_LINES) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-lines:]
    except OSError:
        return []


def build_runtime(tree: str, log_path: str, config: str, cmake_args: list[str]) -> tuple[bool, str | None]:
    """Configure and build gta6rt (and the TLS host DLL it depends on) in <tree>/runtime/build."""
    build = os.path.join(tree, "runtime", "build")
    steps = (["cmake", "-S", os.path.join(tree, "runtime"), "-B", build, *cmake_args],
             ["cmake", "--build", build, "--config", config, "--target", "gta6rt"])
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        for step in steps:
            log.write("$ " + " ".join(step) + "\n")
            log.flush()
            try:
                r = subprocess.run(step, stdout=log, stderr=subprocess.STDOUT)
            except OSError as e:
                log.write(f"cannot start: {e}\n")
                return False, None
            if r.returncode != 0:
                return False, None
    return True, find_gta6rt(build)


def find_gta6rt(build: str) -> str | None:
    for rel in GTA6RT_PATHS:
        path = os.path.join(build, rel)
        if os.path.isfile(path):
            return path
    return None


def stop_tree(proc: subprocess.Popen) -> None:
    """End the process and its children (gta6rt restarts itself suspended to reserve the title's base)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def boot(command: list[str], run_dir: str, env: dict, timeout: float) -> dict:
    trace = os.path.join(run_dir, "boot.jsonl")
    d3d12 = os.path.join(run_dir, "d3d12.log")
    argv = [*command, "--trace", trace, "--trace-d3d12"]
    started = time.monotonic()
    with open(d3d12, "wb") as err, open(os.path.join(run_dir, "stdout.log"), "wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=err, env={**os.environ, **env}, cwd=run_dir)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_tree(proc)
    return {"returncode": None if timed_out else proc.returncode, "timed_out": timed_out,
            "seconds": round(time.monotonic() - started, 1), "trace": trace, "d3d12": d3d12}


def last_d3d12_call(path: str) -> str | None:
    last = None
    try:
        with boot_trace._open_text(path) as fh:
            for line in fh:
                m = boot_trace.D3D12_LINE.match(line.rstrip("\n"))
                if m:
                    last = f"#{m.group(1)} {m.group(2)} {m.group(3)}"
    except boot_trace.TraceError:
        return None
    return last


def classify(startup: dict | None, timed_out: bool, returncode: int | None) -> dict:
    """The stop of one run, from the startup summary and how the process ended."""
    if startup is None:
        return {"kind": "no trace", "returncode": returncode}
    stop = {"exceptions": startup["exception_count"], "unimplemented": len(startup["unimplemented"])}
    last = startup["exceptions"][-1] if startup["exceptions"] else None
    if startup["exit"] is not None:
        stop.update(kind="exit", code=startup["exit"])
    elif timed_out:
        stop.update(kind="running at timeout")
    elif last:
        stop.update(kind="last exception", code=last["code"], at=last["at"], access=last.get("access"),
                    address=last.get("address"), returncode=returncode)
    elif startup["error"]:
        stop.update(kind="error", what=startup["error"], returncode=returncode)
    else:
        stop.update(kind="ended without exit event", returncode=returncode)
    return stop


def summarize_run(outcome: dict) -> dict:
    try:
        startup = boot_trace.parse_startup(outcome["trace"])
    except boot_trace.TraceError:
        startup = None
    try:
        calls = boot_trace.parse_d3d12(outcome["d3d12"])["calls"]
    except boot_trace.TraceError:
        calls = 0
    return {"stop": classify(startup, outcome["timed_out"], outcome["returncode"]), "d3d12_calls": calls,
            "last_d3d12_call": last_d3d12_call(outcome["d3d12"]), "seconds": outcome["seconds"]}


def cmd_run(a: argparse.Namespace) -> int:
    matrix = read_json(os.path.join(a.matrix, MATRIX))
    entry = find_set(matrix, a.set)
    tree = os.path.abspath(a.tree)
    tree_digest = digest_of_tree(tree)
    run_dir = next_run_dir(os.path.join(a.matrix, a.set))
    result = {"set": a.set, "head": entry["head"], "run": os.path.basename(run_dir),
              "translator_matches_set": tree_digest == entry["translation_digest"],
              "shaders": shader_library_state(a.shaders, tree_digest), "build": "skipped"}
    gta6rt = a.gta6rt
    if a.build:
        ok, built = build_runtime(tree, os.path.join(run_dir, "build.log"), a.config, a.cmake_arg)
        result["build"] = "ok" if ok and built else "failed"
        if result["build"] == "failed":
            result["build_tail"] = tail(os.path.join(run_dir, "build.log"))
            write_json(os.path.join(run_dir, RESULT), result)
            print(f"{a.set}: build failed, see {os.path.join(run_dir, 'build.log')}")
            return 1
        gta6rt = built
    if not gta6rt or not os.path.isfile(gta6rt):
        raise MatrixError("give --build or --gta6rt <path to gta6rt.exe>")
    # A fresh overlay and save folder per run: what an earlier run wrote must not change this boot.
    overlay = os.path.abspath(a.overlay) if a.overlay else os.path.join(run_dir, "overlay")
    save = os.path.join(run_dir, "save")
    os.makedirs(overlay, exist_ok=True)
    os.makedirs(save, exist_ok=True)
    env = {"LEONIDA_D3D12": "vulkan", "LEONIDA_SHADERS": os.path.abspath(a.shaders), "LEONIDA_SAVE_DIR": save}
    env.update(dict(a.env))
    command = [os.path.abspath(gta6rt), "--run", os.path.abspath(a.exe), "--game-dir", os.path.abspath(a.game_dir),
               "--overlay", overlay]
    result.update(summarize_run(boot(command, run_dir, env, a.timeout)))
    write_json(os.path.join(run_dir, RESULT), result)
    print(f"{a.set} {result['run']}: {format_stop(result['stop'])}; {result['d3d12_calls']} D3D12 calls"
          f"; shaders {result['shaders']}")
    return 1 if result["stop"]["kind"] == "no trace" else 0


# --- report --------------------------------------------------------------------------------------------------------

def format_stop(stop: dict) -> str:
    kind = stop["kind"]
    if kind == "exit":
        return f"exit {stop['code']}"
    if kind == "last exception":
        text = f"last exception {stop['code']} at {stop['at']}"
        if stop.get("access"):
            text += f" ({stop['access']}" + (f" {stop['address']}" if stop.get("address") else "") + ")"
        return text
    if kind == "error":
        return f"error: {stop['what']}"
    if stop.get("returncode") is not None:
        return f"{kind} (process exit {stop['returncode']})"
    return kind


def latest_result(matrix_dir: str, name: str) -> dict | None:
    set_dir = os.path.join(matrix_dir, name)
    runs = []
    if os.path.isdir(set_dir):
        for entry in os.listdir(set_dir):
            m = re.fullmatch(r"run(\d+)", entry)
            if m and os.path.isfile(os.path.join(set_dir, entry, RESULT)):
                runs.append((int(m.group(1)), entry))
    if not runs:
        return None
    return read_json(os.path.join(set_dir, max(runs)[1], RESULT))


def skip_reason(pr: dict) -> str:
    """Why a PR is not in the set's head: already on the base, or which merge it conflicts with."""
    if pr["status"] != "conflict":
        return pr["status"]
    return "conflicts with base" if pr.get("conflicts_with_base") else "conflicts with an earlier PR"


def pr_cell(prs: list[dict], status: str) -> str:
    return " ".join(f"#{p['pr']}" for p in prs if p["status"] == status) or "-"


def report_rows(matrix_dir: str) -> tuple[dict, list[dict]]:
    matrix = read_json(os.path.join(matrix_dir, MATRIX))
    rows = []
    for s in matrix["sets"]:
        rows.append({"set": s["name"], "head": s["head"], "prs": s["prs"], "result": latest_result(matrix_dir, s["name"])})
    return matrix, rows


def format_markdown(matrix: dict, rows: list[dict]) -> str:
    out = [f"Base `{matrix['base']['sha'][:12]}` ({matrix['base']['ref']}).", "",
           "| Set | Head | Merged | Skipped | Build | Shaders | D3D12 calls | Last D3D12 call | Stop |",
           "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        skipped = " ".join(f"#{p['pr']} ({skip_reason(p)})" for p in row["prs"] if p["status"] != "merged") or "-"
        r = row["result"]
        if r is None:
            run = ["not run"] + ["-"] * 4
        else:
            last = f"`{r['last_d3d12_call']}`" if r.get("last_d3d12_call") else "-"
            stop = format_stop(r["stop"]) if "stop" in r else "-"
            run = [r["build"], r["shaders"], str(r.get("d3d12_calls", "-")), last, stop]
        out.append("| " + " | ".join([row["set"], f"`{row['head'][:12]}`", pr_cell(row["prs"], "merged"), skipped,
                                       *run]) + " |")
    return "\n".join(out)


def cmd_report(a: argparse.Namespace) -> int:
    matrix, rows = report_rows(a.matrix)
    if a.json:
        print(json.dumps({"base": matrix["base"], "rows": rows}, indent=1))
    else:
        print(format_markdown(matrix, rows))
    return 0


# --- command line --------------------------------------------------------------------------------------------------

def parse_env(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("merge", help="merge each set onto the base (git only)")
    m.add_argument("--base", required=True, help="the commit every set starts from, e.g. origin/main")
    m.add_argument("--set", action="append", required=True, type=parse_set, metavar="NAME=PR[,PR...]",
                   help="a named set, merged in the order given; a PR is N or N:REF (repeat for more sets)")
    m.add_argument("--fetch", metavar="REMOTE", help="fetch pull/N/head for each PR given as N from this remote")
    m.add_argument("--repo", default=".", help="repository folder (default: the current one)")
    m.add_argument("--out", required=True, help="the matrix folder (matrix.json, later the runs)")
    m.add_argument("--min-length", type=int, default=40, help="merge_check: shortest duplicate line counted (40)")
    m.add_argument("--depth", type=int, default=5000, help="merge_check: first-parent commits searched (5000)")
    m.set_defaults(func=cmd_merge)

    r = sub.add_parser("run", help="build and boot one set's tree")
    r.add_argument("matrix", help="the matrix folder written by merge")
    r.add_argument("--set", required=True, help="the set's name")
    r.add_argument("--tree", required=True, help="a checkout or export of the set's branch")
    r.add_argument("--exe", required=True, help="game_scarlett_master.exe")
    r.add_argument("--game-dir", required=True, help="the package folder")
    r.add_argument("--shaders", required=True, help="<work>/shaders/spirv, passed as LEONIDA_SHADERS")
    r.add_argument("--build", action="store_true", help="configure and build runtime/ in the tree first")
    r.add_argument("--config", default="RelWithDebInfo", help="build configuration (RelWithDebInfo)")
    r.add_argument("--cmake-arg", action="append", default=[], metavar="ARG",
                   help="extra configure argument for --build, e.g. -GNinja or -DLEONIDA_VULKAN_HEADERS_FETCH=OFF "
                        "on a machine without network (repeatable; use --cmake-arg=-G... for a leading dash)")
    r.add_argument("--gta6rt", help="an already built gta6rt.exe (instead of --build)")
    r.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"seconds before the run is stopped "
                   f"({DEFAULT_TIMEOUT})")
    r.add_argument("--overlay", help="gta6rt's --overlay (default: a fresh <run folder>/overlay)")
    r.add_argument("--env", action="append", default=[], type=parse_env, metavar="NAME=VALUE",
                   help="extra environment for gta6rt, e.g. LEONIDA_SKIP_INT3=1 (repeatable)")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="the newest run of each set as a Markdown table")
    p.add_argument("matrix", help="the matrix folder")
    p.add_argument("--json", action="store_true", help="JSON instead of Markdown")
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "timeout", 1) <= 0:
        print("boot_matrix: --timeout must be positive", file=sys.stderr)
        return 2
    if shutil.which("git") is None and a.command == "merge":
        print("boot_matrix: git is not on PATH", file=sys.stderr)
        return 2
    try:
        return a.func(a)
    except MatrixError as e:
        print(f"boot_matrix: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
