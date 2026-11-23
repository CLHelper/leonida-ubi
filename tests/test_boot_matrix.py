"""tools/boot_matrix.py on a throwaway git repository and a stand-in for gta6rt (no network, no game data).

    python -I tests/test_boot_matrix.py

The merge cases build a base and pull request branches, one of them already merged into the base and one that
conflicts with an earlier PR in the set, and check the statuses, the branch and that the working tree is untouched.
The run cases boot a small Python script that writes synthetic startup and D3D12 traces the way gta6rt does: an exit,
a crash after an access violation, and a run that is still going when the timeout stops it.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import boot_matrix as bm  # noqa: E402

MAKE_GAME = 'TRANSLATION_SOURCES = ("rdna2spirv.py", "shader_ir.py")\n'

FAKE_GTA6RT = textwrap.dedent('''\
    import json, sys, time
    args = sys.argv[1:]
    mode = args[0]
    trace = args[args.index("--trace") + 1]
    events = [{"kind": "trace_open", "pid": 1}, {"kind": "map", "module": "title.exe", "base": "0x140000000"},
              {"kind": "unimplemented_call", "import": "a.dll!F", "caller": "title.exe+0x10"}]
    if mode == "crash":
        events.append({"kind": "exception", "code": "0xc0000005", "at": "title.exe+0x4fb0aec",
                       "access": "read", "address": "0xffffffffffffffff"})
    with open(trace, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\\n")
        f.flush()
        for seq, method in enumerate(["CreateCommandQueue", "CreateFence", "ExecuteCommandLists"], 1):
            sys.stderr.write(f"[d3d12-trace] #{seq} {method} ok\\n")
        sys.stderr.write("[rt] d3d12: a diagnostic line\\n")
        sys.stderr.flush()
        if mode == "exit":
            f.write(json.dumps({"kind": "exit", "code": 0}) + "\\n")
        elif mode == "hang":
            f.flush()
            time.sleep(60)
    sys.exit(3 if mode == "crash" else 0)
    ''')


def clean_git_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
    return env


class RepoCase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida boot_matrix ")
        self.addCleanup(td.cleanup)
        self.tmp = td.name
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(self.repo)
        patcher = mock.patch.dict(os.environ, clean_git_env(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.git("init", "-q", "-b", "main")
        self.write("make_game.py", MAKE_GAME)
        self.write("tools/rdna2spirv.py", "# translator v1\n")
        self.write("tools/shader_ir.py", "# ir\n")
        self.write("runtime/slots.txt", "slot 10: list proxy\nslot 35: queue\nslot 108: allocation info\n")
        self.commit("start")

    def git(self, *args):
        r = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "-c", "commit.gpgsign=false", "-C", self.repo, *args],
                           capture_output=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def write(self, path, text):
        full = os.path.join(self.repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def pr(self, number, path, text, base="main"):
        """A PR branch from base that writes path; its head is refs/boot-matrix/pr/N, as --fetch leaves it."""
        self.git("checkout", "-q", "-b", f"pr{number}", base)
        self.write(path, text)
        sha = self.commit(f"PR {number}")
        self.git("checkout", "-q", "main")
        self.git("update-ref", bm.PR_REF.format(number), sha)
        return sha

    def merge(self, *sets, expect=0):
        out = os.path.join(self.tmp, "matrix")
        args = ["merge", "--repo", self.repo, "--base", "main", "--out", out]
        for s in sets:
            args += ["--set", s]
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(bm.main(args), expect)
        with open(os.path.join(out, bm.MATRIX), encoding="utf-8") as f:
            return out, json.load(f)


class ParseTest(unittest.TestCase):
    def test_set_spec(self):
        name, prs = bm.parse_set("rt5=7286,#8517:refs/heads/x")
        self.assertEqual(name, "rt5")
        self.assertEqual(prs, [{"pr": 7286, "ref": "refs/boot-matrix/pr/7286"}, {"pr": 8517, "ref": "refs/heads/x"}])

    def test_base_alone(self):
        self.assertEqual(bm.parse_set("main="), ("main", []))

    def test_bad_specs(self):
        for text in ("rt5", "=1", "a b=1", "rt5=x", "rt5=1,1", "rt5=1,"):
            with self.subTest(text=text), self.assertRaises(Exception):
                bm.parse_set(text)

    def test_translation_sources_read_without_running(self):
        text = 'import os\nTRANSLATION_SOURCES = ("a.py",\n    "b.json")\nraise SystemExit("must not run")\n'
        self.assertEqual(bm.translation_sources(text), ("a.py", "b.json"))
        with self.assertRaises(bm.MatrixError):
            bm.translation_sources("X = 1\n")


class DigestTest(unittest.TestCase):
    def test_matches_make_game(self):
        """The digest over this checkout equals the one make_game.py writes beside the shader library."""
        sys.path.insert(0, ROOT)
        import make_game  # noqa: E402
        self.assertEqual(bm.digest_of_tree(ROOT), make_game.translation_sources_digest())


class MergeTest(RepoCase):
    def test_statuses_branch_and_untouched_worktree(self):
        base = self.git("rev-parse", "main")
        a = self.pr(1, "runtime/slot10.txt", "unwrap list proxies\n")
        self.pr(2, "runtime/slots.txt", "slot 10: bypass\nslot 35: queue\nslot 108: allocation info\n")
        self.pr(3, "runtime/slots.txt", "slot 10: unwrap\nslot 35: queue\nslot 108: allocation info\n")
        fixed = self.pr(4, "runtime/slot108.txt", "return buffer first\n")
        self.git("merge", "-q", "--no-ff", "-m", "Merge pull request #4 from someone/slot108", fixed)
        base = self.git("rev-parse", "main")
        status_before = self.git("status", "--porcelain")

        out, matrix = self.merge("stack=1,2,3,4", "small=1", expect=1)

        self.assertEqual(matrix["base"]["sha"], base)
        stack = matrix["sets"][0]
        self.assertEqual([(p["pr"], p["status"]) for p in stack["prs"]],
                         [(1, "merged"), (2, "merged"), (3, "conflict"), (4, "already-merged")])
        self.assertEqual(stack["prs"][2]["conflict_files"], ["runtime/slots.txt"])
        self.assertEqual(self.git("rev-parse", "refs/heads/boot-matrix/stack"), stack["head"])
        # Two merge commits on the base: first parents walk back to it, second parents are the PRs.
        parents = self.git("log", "--first-parent", "--format=%P", f"{base}..{stack['head']}").splitlines()
        self.assertEqual(len(parents), 2)
        self.assertEqual(parents[-1].split(), [base, a])
        self.assertEqual(self.git("show", f"{stack['head']}:runtime/slots.txt").splitlines()[0], "slot 10: bypass")
        self.assertEqual(self.git("rev-parse", "HEAD"), base)
        self.assertEqual(self.git("status", "--porcelain"), status_before)
        self.assertEqual(matrix["sets"][1]["prs"][0]["status"], "merged")
        # #3 merges cleanly onto the base alone; it is skipped because of #2 before it.
        self.assertEqual(bm.skip_reason(stack["prs"][2]), "conflicts with an earlier PR")
        self.assertEqual(bm.skip_reason(stack["prs"][3]), "already-merged")

    def test_conflict_with_base_and_base_alone(self):
        self.pr(1, "runtime/slots.txt", "slot 10: bypass\n")
        self.write("runtime/slots.txt", "slot 10: moved on main\n")
        base = self.commit("main moves on")
        _, matrix = self.merge("control=", "s=1", expect=1)
        self.assertEqual((matrix["sets"][0]["head"], matrix["sets"][0]["prs"]), (base, []))
        self.assertEqual(self.git("rev-parse", "refs/heads/boot-matrix/control"), base)
        self.assertEqual(bm.skip_reason(matrix["sets"][1]["prs"][0]), "conflicts with base")

    def test_same_inputs_same_commit(self):
        self.pr(1, "runtime/slot10.txt", "unwrap\n")
        _, first = self.merge("s=1")
        _, second = self.merge("s=1")
        self.assertEqual(first["sets"][0]["head"], second["sets"][0]["head"])
        author = self.git("show", "-s", "--format=%an <%ae>", first["sets"][0]["head"])
        self.assertEqual(author, "boot_matrix <boot-matrix@invalid>")

    def test_translator_change_changes_digest(self):
        self.pr(1, "tools/rdna2spirv.py", "# translator v2\n")
        self.pr(2, "runtime/slot10.txt", "unwrap\n")
        _, matrix = self.merge("translator=1", "runtime=2")
        base_digest = bm.digest_of_commit(self.repo, matrix["base"]["sha"])
        self.assertNotEqual(matrix["sets"][0]["translation_digest"], base_digest)
        self.assertEqual(matrix["sets"][1]["translation_digest"], base_digest)
        self.assertEqual(bm.digest_of_tree(self.repo), base_digest)

    def test_missing_ref_is_a_usage_error(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            code = bm.main(["merge", "--repo", self.repo, "--base", "main", "--out",
                            os.path.join(self.tmp, "m"), "--set", "s=99"])
        self.assertEqual(code, 2)
        self.assertIn("--fetch", err.getvalue())


class RunTest(RepoCase):
    def setUp(self):
        super().setUp()
        self.pr(1, "runtime/slot10.txt", "unwrap\n")
        self.out, self.matrix = self.merge("s=1")
        self.fake = os.path.join(self.tmp, "fake_gta6rt.py")
        with open(self.fake, "w", encoding="utf-8", newline="\n") as f:
            f.write(FAKE_GTA6RT)
        self.shaders = os.path.join(self.tmp, "spirv")
        os.makedirs(self.shaders)

    def record_shaders(self, digest):
        with open(os.path.join(self.shaders, "translation_sources.sha256"), "w", encoding="utf-8") as f:
            f.write(digest + "\n")

    def run_set(self, mode, timeout=30, expect=0):
        real_boot = bm.boot
        seen = {}

        def fake_boot(command, run_dir, env, timeout_s):
            seen["env"] = env
            seen["command"] = command
            return real_boot([sys.executable, "-I", self.fake, mode], run_dir, env, timeout_s)

        gta6rt = os.path.join(self.tmp, "gta6rt.exe")
        open(gta6rt, "wb").close()
        args = ["run", self.out, "--set", "s", "--tree", self.repo, "--exe", "title.exe", "--game-dir", "pkg",
                "--shaders", self.shaders, "--gta6rt", gta6rt, "--timeout", str(timeout),
                "--env", "LEONIDA_SKIP_INT3=1"]
        with mock.patch.object(bm, "boot", fake_boot), redirect_stdout(io.StringIO()):
            self.assertEqual(bm.main(args), expect)
        return bm.latest_result(self.out, "s"), seen

    def test_exit(self):
        self.record_shaders(bm.digest_of_tree(self.repo))
        result, seen = self.run_set("exit")
        self.assertEqual(result["stop"]["kind"], "exit")
        self.assertEqual(result["stop"]["code"], 0)
        self.assertEqual(result["d3d12_calls"], 3)
        self.assertEqual(result["last_d3d12_call"], "#3 ExecuteCommandLists ok")
        self.assertEqual(result["shaders"], "current")
        self.assertEqual(seen["env"]["LEONIDA_D3D12"], "vulkan")
        self.assertEqual(seen["env"]["LEONIDA_SKIP_INT3"], "1")
        self.assertEqual(seen["command"][1:], ["--run", "title.exe", "--game-dir", "pkg"])

    def test_crash_names_the_last_exception(self):
        self.record_shaders("0" * 64)
        result, _ = self.run_set("crash")
        stop = result["stop"]
        self.assertEqual((stop["kind"], stop["code"], stop["at"]), ("last exception", "0xc0000005",
                                                                     "title.exe+0x4fb0aec"))
        self.assertEqual(stop["returncode"], 3)
        self.assertEqual(result["shaders"], "stale")
        self.assertEqual(bm.format_stop(stop),
                         "last exception 0xc0000005 at title.exe+0x4fb0aec (read 0xffffffffffffffff)")

    def test_timeout_stops_the_run(self):
        result, _ = self.run_set("hang", timeout=3)
        self.assertEqual(result["stop"]["kind"], "running at timeout")
        self.assertLess(result["seconds"], 30)
        self.assertEqual(result["shaders"], "unknown")

    def test_runs_are_numbered_and_report_uses_the_newest(self):
        self.run_set("crash")
        self.run_set("exit")
        self.assertTrue(os.path.isdir(os.path.join(self.out, "s", "run2")))
        matrix, rows = bm.report_rows(self.out)
        text = bm.format_markdown(matrix, rows)
        self.assertIn("| s | `" + self.matrix["sets"][0]["head"][:12] + "` | #1 | - | skipped | unknown | 3 | "
                      "`#3 ExecuteCommandLists ok` | exit 0 |", text)

    def test_no_trace(self):
        stop = bm.classify(None, False, 1)
        self.assertEqual(bm.format_stop(stop), "no trace (process exit 1)")


class ReportTest(RepoCase):
    def test_not_run_and_skipped(self):
        self.pr(1, "runtime/slots.txt", "a\n")
        self.pr(2, "runtime/slots.txt", "b\n")
        out, _ = self.merge("s=1,2", expect=1)
        matrix, rows = bm.report_rows(out)
        text = bm.format_markdown(matrix, rows)
        self.assertIn("| #1 | #2 (conflicts with an earlier PR) | not run | - | - | - | - |", text)


if __name__ == "__main__":
    unittest.main()
