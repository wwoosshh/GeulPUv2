#!/usr/bin/env python3
"""Experiment check — does an experiment folder follow GeulPU's experiment record format (geulpu-experiment/1)?

An experiment is one folder under experiments/ (experiments/README.md). Standard library only, no GPU. What it checks:

- experiment.json against experiments/experiment.schema.json, the one definition of the format. The checker refuses a schema that uses a
  JSON Schema keyword it does not implement, so the schema cannot promise more than is checked.
- `id` is the folder name; plan.md exists, and results.md when `status` is `done`.
- Every file in the folder, except the three documents (experiment.json · plan.md · results.md), is listed in `files` with the right
  SHA-256 and size, and every listed file is in the folder.
- Run ids are unique, a branch's `base` is a run, every run `record` is listed in `files`, every run in expected.json is in `runs`.
- The folder compressed with zip (deflate) is at most 5 MB.
- experiments/index.md links every experiment folder (not _template), and every experiment folder it links exists.

    python 도구/experiment_check.py                                # every folder in experiments/, and index.md
    python 도구/experiment_check.py experiments/<folder> …         # these folders
    python 도구/experiment_check.py --files experiments/<folder>   # print the `files` list to paste into experiment.json
    python 도구/experiment_check.py --self-test                    # break copies of a correct experiment; every fault must be caught

Exit code 0 when everything passes, 1 when something fails, 2 on a usage error.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.join(ROOT, "experiments")
# the GeulPU app ships this checker with the schema next to it (data/ — `GeulPU experiment check`); in the repository it is experiments/
SCHEMA = os.path.join(HERE, "experiment.schema.json")
if not os.path.isfile(SCHEMA):
    SCHEMA = os.path.join(EXPERIMENTS, "experiment.schema.json")
TEMPLATE = "_template"
DOCUMENTS = ("experiment.json", "plan.md", "results.md")      # the three documents — every other file goes in `files`
LIMIT = 5 * 1024 * 1024                                        # one experiment, compressed
SKIP_DIRS = ("__pycache__",)
EXPERIMENT_ID = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-")
LINK = re.compile(r"\]\(\s*<?(?:\./)?([^)\s>]+)")             # the target of a Markdown link


# ---- JSON Schema: the subset experiment.schema.json uses ----
TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}
KEYWORDS = {"$schema", "$id", "title", "description", "type", "required", "properties", "const", "enum", "pattern", "minLength",
            "minItems", "minimum", "items"}


def unknown_keywords(schema, where="#"):
    """The keywords in schema that validate() does not implement."""
    bad = []
    for k, v in schema.items():
        if k not in KEYWORDS:
            bad.append(f"{where}/{k}")
        elif k == "type" and (not isinstance(v, str) or v not in TYPES):
            bad.append(f"{where}/type {json.dumps(v)}")
        elif k == "properties":
            for name, sub in v.items():
                bad += unknown_keywords(sub, f"{where}/properties/{name}")
        elif k == "items":
            bad += unknown_keywords(v, f"{where}/items")
    return bad


def validate(value, schema, where, errors):
    """Append to errors every way value breaks schema; where is the field path ('' at the top)."""
    name = where or "experiment.json"
    if "const" in schema and value != schema["const"]:
        errors.append(f"{name} must be {json.dumps(schema['const'])}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{name} must be one of " + " · ".join(json.dumps(e) for e in schema["enum"]))
    t = schema.get("type")
    if t and not TYPES[t](value):
        errors.append(f"{name} must be of type {t}")
        return
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{name} must not be empty" if schema["minLength"] == 1 else f"{name} is shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{name} {json.dumps(value, ensure_ascii=False)} does not match {schema['pattern']}")
    if TYPES["number"](value) and "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{name} must be at least {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{name} needs at least {schema['minItems']} item" + ("s" if schema["minItems"] != 1 else ""))
        if "items" in schema:
            for i, v in enumerate(value):
                validate(v, schema["items"], f"{where}[{i}]", errors)
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{name}: required field '{k}' is missing")
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                validate(value[k], sub, f"{where}.{k}" if where else k, errors)


# ---- files ----
def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def sha256_file(path):
    """(SHA-256 in hex, size in bytes) of a file."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def folder_files(folder):
    """Every file in the folder, as relative paths with '/', sorted."""
    out = []
    for d, dirs, names in os.walk(folder):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        out += [os.path.relpath(os.path.join(d, n), folder).replace(os.sep, "/") for n in names]
    return sorted(out)


def list_files(folder):
    """The `files` list of a folder: every file but the three documents, with SHA-256 and size."""
    out = []
    for rel in folder_files(folder):
        if rel not in DOCUMENTS:
            digest, size = sha256_file(os.path.join(folder, *rel.split("/")))
            out.append({"path": rel, "sha256": digest, "bytes": size})
    return out


class _Counter:
    """A write-only stream that counts bytes; zipfile writes to such a stream with data descriptors."""

    def __init__(self):
        self.bytes = 0

    def write(self, b):
        self.bytes += len(b)
        return len(b)

    def flush(self):
        pass

    def close(self):
        pass


def zipped_size(folder, rels):
    """The size of the folder as a zip file (deflate, default level)."""
    c = _Counter()
    with zipfile.ZipFile(c, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
        for rel in rels:
            z.write(os.path.join(folder, *rel.split("/")), rel)
    return c.bytes


# ---- the checks ----
def check_folder(folder, schema):
    """The errors of one experiment folder — an empty list when it passes."""
    name = os.path.basename(os.path.normpath(folder))
    if not os.path.isfile(os.path.join(folder, "experiment.json")):
        return ["experiment.json is missing"]
    try:
        exp = load_json(os.path.join(folder, "experiment.json"))
    except (ValueError, UnicodeDecodeError) as e:
        return [f"experiment.json is not valid JSON — {e}"]
    errors = []
    validate(exp, schema, "", errors)
    if not isinstance(exp, dict):
        return errors
    if exp.get("id") != name:
        errors.append(f"id {json.dumps(exp.get('id'), ensure_ascii=False)} is not the folder name {json.dumps(name, ensure_ascii=False)}")
    if not os.path.isfile(os.path.join(folder, "plan.md")):
        errors.append("plan.md is missing")
    if exp.get("status") == "done" and not os.path.isfile(os.path.join(folder, "results.md")):
        errors.append("status is done but results.md is missing")

    # every file in the folder is listed, with its SHA-256 and size
    present = folder_files(folder)
    listed = {}
    for i, f in enumerate(exp["files"] if isinstance(exp.get("files"), list) else []):
        rel = f.get("path") if isinstance(f, dict) else None
        if not isinstance(rel, str):
            continue                                          # the schema check reported it
        parts = rel.split("/")
        if "\\" in rel or ":" in rel or any(p in ("", ".", "..") for p in parts):
            errors.append(f"files[{i}].path {rel} must be a relative path inside the folder, with '/'")
            continue
        if rel in DOCUMENTS:
            errors.append(f"files: {rel} is one of the three documents, which are not listed")
            continue
        if rel in listed:
            errors.append(f"files: {rel} is listed twice")
            continue
        listed[rel] = f
        full = os.path.join(folder, *parts)
        if not os.path.isfile(full):
            errors.append(f"files: {rel} is listed but not in the folder")
            continue
        digest, size = sha256_file(full)
        if f.get("bytes") != size:
            errors.append(f"files: {rel} has {size} bytes, listed as {f.get('bytes')}")
        if f.get("sha256") != digest:
            errors.append(f"files: the SHA-256 of {rel} is {digest[:16]}…, listed as {str(f.get('sha256'))[:16]}…")
    for rel in present:
        if rel not in DOCUMENTS and rel not in listed:
            errors.append(f"{rel} is in the folder but not in files")

    # runs: unique ids, known bases, listed records; expected.json names only runs
    runs = [r for r in exp["runs"] if isinstance(r, dict)] if isinstance(exp.get("runs"), list) else []
    ids = [r["id"] for r in runs if isinstance(r.get("id"), str)]
    errors += [f"runs: id {rid} is used more than once" for rid in sorted({x for x in ids if ids.count(x) > 1})]
    for r in runs:
        record, base = r.get("record"), r.get("base")
        if isinstance(record, str) and record not in listed:
            errors.append(f"runs: the record {record} of run {r.get('id')} is not in files")
        if isinstance(base, str) and base not in ids:
            errors.append(f"runs: run {r.get('id')} branches from {base}, which is not a run")
    if os.path.isfile(os.path.join(folder, "expected.json")):
        try:
            expected = load_json(os.path.join(folder, "expected.json"))
        except (ValueError, UnicodeDecodeError) as e:
            errors.append(f"expected.json is not valid JSON — {e}")
        else:
            er = expected.get("runs") if isinstance(expected, dict) else None
            if not isinstance(er, dict):
                errors.append('expected.json must be {"runs": {"<run id>": {...}}}')
            else:
                errors += [f"expected.json: run {rid} is not in runs" for rid in er if rid not in ids]

    size = zipped_size(folder, present)
    if size > LIMIT:
        errors.append(f"the folder compresses to {size / 1048576:.2f} MB, more than 5 MB — keep large raw data outside the repository "
                      "and point to it by SHA-256")
    return errors


def index_errors(root, names, whole):
    """index.md in root must link every experiment folder in names (not _template); with whole, every experiment folder it links must
    exist in root."""
    path = os.path.join(root, "index.md")
    if not os.path.isfile(path):
        return ["index.md is missing"]
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    linked = set()
    for m in LINK.finditer(text):
        first = m.group(1).split("#")[0].split("/")[0]
        if EXPERIMENT_ID.match(first):
            linked.add(first)
    errors = [f"index.md does not link {n}/" for n in names if n != TEMPLATE and n not in linked]
    if whole:
        errors += [f"index.md links {n}/, which is not a folder here" for n in sorted(linked) if not os.path.isdir(os.path.join(root, n))]
    return errors


def experiment_folders(root):
    return sorted(n for n in os.listdir(root)
                  if os.path.isdir(os.path.join(root, n)) and n not in SKIP_DIRS and not n.startswith("."))


def load_schema():
    schema = load_json(SCHEMA)
    bad = unknown_keywords(schema)
    if bad:
        sys.exit("experiment.schema.json uses JSON Schema keywords this checker does not implement (add them to "
                 "도구/experiment_check.py first): " + ", ".join(bad))
    return schema


def shown(path):
    try:
        rel = os.path.relpath(os.path.abspath(path), ROOT)
    except ValueError:                                        # another drive
        return path
    return path if rel.startswith("..") else rel.replace(os.sep, "/")


def report(label, errors):
    print(("FAIL " if errors else "pass ") + label)
    for e in errors:
        print("     " + e)
    return 1 if errors else 0


def check(folders, whole, as_json=False):
    """Check the folders and their index.md. Text: one line per folder (with its errors) and a summary; with as_json one JSON object
    {"format", "pass", "summary", "results": [{"path", "kind", "pass", "problems"}]} (the app's `experiment check` reads it). The exit code."""
    schema = load_schema()
    results = [{"path": shown(folder), "kind": "folder", "problems": check_folder(folder, schema)} for folder in folders]
    by_root = {EXPERIMENTS: []} if whole else {}
    for folder in folders:
        full = os.path.normpath(os.path.abspath(folder))
        by_root.setdefault(os.path.dirname(full), []).append(os.path.basename(full))
    for root, names in by_root.items():
        if whole or any(n != TEMPLATE for n in names):
            results.append({"path": shown(os.path.join(root, "index.md")), "kind": "index", "problems": index_errors(root, names, whole)})
    n = len(folders)
    failed = sum(1 for r in results if r["kind"] == "folder" and r["problems"])
    indexes = [r for r in results if r["kind"] == "index"]
    line = f"experiment check: folders {n - failed}/{n} pass"
    if indexes:
        line += " · index.md " + ("fails" if any(r["problems"] for r in indexes) else "passes")
    ok = not any(r["problems"] for r in results)
    if as_json:
        print(json.dumps({"format": "geulpu-experiment-check/1", "pass": ok, "summary": line,
                          "results": [{**r, "pass": not r["problems"]} for r in results]}, ensure_ascii=False))
    else:
        for r in results:
            report(r["path"], r["problems"])
        print(line)
    return 0 if ok else 1


# ---- self-test: break copies of a correct experiment ----
SELF_TEST_ID = "2026-11-08-self-test"


def _write(folder, rel, text):
    path = os.path.join(folder, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
        f.write("\n")


def _edit(folder, change, relist=False):
    """change(experiment) on experiment.json; with relist, list the files again afterwards."""
    path = os.path.join(folder, "experiment.json")
    exp = load_json(path)
    change(exp)
    if relist:
        exp["files"] = list_files(folder)
    _write_json(path, exp)


def _index(root, *ids):
    _write(root, "index.md", "# Experiments\n\n| Experiment | Title |\n|---|---|\n" + "".join(f"| [{i}]({i}/) | test |\n" for i in ids))


def _good_experiment(root):
    """A correct, finished experiment made from _template in root, linked from root/index.md. Its folder."""
    folder = os.path.join(root, SELF_TEST_ID)
    shutil.copytree(os.path.join(EXPERIMENTS, TEMPLATE), folder)
    _write(folder, "records/r1.jsonl", "".join(json.dumps({"step": i, "token": 100 + i}) + "\n" for i in range(8)))
    _write_json(os.path.join(folder, "expected.json"), {"runs": {"r1": {"tokens_sha256": "0" * 64}, "r2": {"tokens_sha256": "1" * 64}}})

    def finish(exp):
        exp.update(id=SELF_TEST_ID, status="done")
        exp["models"] = [{"name": "test model", "sha256": "a" * 64, "source": {"local": "self-test"}}]
        exp["runs"] = [{"id": "r1", "kind": "generate", "operation": "run.complete", "record": "records/r1.jsonl", "seed": 1},
                       {"id": "r2", "kind": "branch", "operation": "run.branch", "base": "r1",
                        "intervention": {"type": "token", "step": 3, "value": 7}}]
        exp["criteria"][0]["result"] = "pass"
    _edit(folder, finish, relist=True)
    _index(root, SELF_TEST_ID)
    return folder


def _flip_first_byte(folder):
    with open(os.path.join(folder, "records", "r1.jsonl"), "r+b") as f:
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 1]))


def _add_big_file(folder):
    with open(os.path.join(folder, "records", "big.bin"), "wb") as f:
        f.write(random.Random(0).randbytes(LIMIT + 256 * 1024))     # does not compress
    _edit(folder, lambda e: None, relist=True)


def _expect_unknown_run(folder):
    _write_json(os.path.join(folder, "expected.json"), {"runs": {"r1": {}, "r9": {}}})
    _edit(folder, lambda e: None, relist=True)


# (fault, break the copy (folder, root), text one of the errors must contain). The first three are the faults named in the criteria
# of the format (문서/기능목록.md §6).
FAULTS = [
    ("included file changed (hash differs)", lambda d, r: _flip_first_byte(d), "SHA-256 of records/r1.jsonl"),
    ("folder over 5 MB compressed", lambda d, r: _add_big_file(d), "compresses to"),
    ("required field missing (question)", lambda d, r: _edit(d, lambda e: e.pop("question")), "required field 'question'"),
    ("status done without results.md", lambda d, r: os.remove(os.path.join(d, "results.md")), "results.md is missing"),
    ("no criteria", lambda d, r: _edit(d, lambda e: e.update(criteria=[])), "criteria needs at least 1 item"),
    ("experiment folder not in index.md", lambda d, r: _index(r), f"index.md does not link {SELF_TEST_ID}/"),
    ("id is not the folder name", lambda d, r: _edit(d, lambda e: e.update(id="2026-11-08-other")), "is not the folder name"),
    ("plan.md missing", lambda d, r: os.remove(os.path.join(d, "plan.md")), "plan.md is missing"),
    ("wrong format", lambda d, r: _edit(d, lambda e: e.update(format="geulpu-experiment/2")), "format must be"),
    ("unknown status", lambda d, r: _edit(d, lambda e: e.update(status="finished")), "status must be one of"),
    ("model hash is not a SHA-256", lambda d, r: _edit(d, lambda e: e["models"][0].update(sha256="abc")), "models[0].sha256"),
    ("file in the folder but not listed", lambda d, r: _write(d, "records/extra.txt", "x"), "records/extra.txt is in the folder but not"),
    ("listed file missing", lambda d, r: os.remove(os.path.join(d, "records", "r1.jsonl")), "is listed but not in the folder"),
    ("listed size wrong", lambda d, r: _edit(d, lambda e: e["files"][0].update(bytes=e["files"][0]["bytes"] + 1)), "bytes, listed as"),
    ("listed path outside the folder",
     lambda d, r: _edit(d, lambda e: e["files"].append({"path": "../x", "sha256": "0" * 64, "bytes": 0})), "inside the folder"),
    ("a document listed in files",
     lambda d, r: _edit(d, lambda e: e["files"].append({"path": "plan.md", "sha256": "0" * 64, "bytes": 0})), "three documents"),
    ("run record not listed", lambda d, r: _edit(d, lambda e: e["runs"][0].update(record="records/r9.jsonl")), "record records/r9.jsonl"),
    ("run id used twice", lambda d, r: _edit(d, lambda e: e["runs"][1].update(id="r1")), "used more than once"),
    ("branch from an unknown run", lambda d, r: _edit(d, lambda e: e["runs"][1].update(base="r9")), "branches from r9"),
    ("expected.json names an unknown run", lambda d, r: _expect_unknown_run(d), "run r9 is not in runs"),
    ("index.md links a missing folder", lambda d, r: _index(r, SELF_TEST_ID, "2026-01-01-gone"), "2026-01-01-gone/, which is not a folder"),
]


def self_test():
    t0 = time.time()
    schema = load_schema()
    tmp = tempfile.mkdtemp(prefix="experiment_check_")
    try:
        def fresh(k):
            root = os.path.join(tmp, str(k), "experiments")
            os.makedirs(root)
            return root, _good_experiment(root)

        def errors_of(root, folder):
            return check_folder(folder, schema) + index_errors(root, [SELF_TEST_ID], True)

        template = check_folder(os.path.join(EXPERIMENTS, TEMPLATE), schema)
        report("the template passes", template)
        control = errors_of(*fresh(0))
        report("control: a correct experiment passes", control)
        caught = 0
        for k, (fault, breaks, marker) in enumerate(FAULTS, 1):
            root, folder = fresh(k)
            breaks(folder, root)
            errors = errors_of(root, folder)
            hit = [e for e in errors if marker in e]
            caught += bool(hit)
            print(f"{'caught' if hit else 'MISSED'} {fault} — {hit[0] if hit else ('; '.join(errors) or 'no error')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = not template and not control and caught == len(FAULTS)
    print(f"self-test: template {'passes' if not template else 'FAILS'} · control {'passes' if not control else 'FAILS'} · "
          f"{caught}/{len(FAULTS)} faults caught ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description="Check GeulPU experiment records (experiments/README.md). Standard library only, no GPU.")
    p.add_argument("folders", nargs="*", metavar="FOLDER", help="experiment folders (default: every folder in experiments/, and index.md)")
    p.add_argument("--files", metavar="FOLDER", help="print the files list of FOLDER: every file but the three documents, with SHA-256 "
                                                    "and size")
    p.add_argument("--self-test", action="store_true", help="break copies of a correct experiment in every checked way; each fault must "
                                                            "be caught")
    p.add_argument("--json", action="store_true", help="print one JSON object (pass, summary, results) instead of text")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if a.files:
        if not os.path.isdir(a.files):
            print(f"not a folder: {a.files}", file=sys.stderr)
            return 2
        entries = list_files(a.files)
        print("[" + ",".join("\n  " + json.dumps(e, ensure_ascii=False) for e in entries) + ("\n" if entries else "") + "]")
        return 0
    if a.folders:
        missing = [f for f in a.folders if not os.path.isdir(f)]
        if missing:
            print("not a folder: " + ", ".join(missing), file=sys.stderr)
            return 2
        return check(a.folders, whole=False, as_json=a.json)
    if not os.path.isdir(EXPERIMENTS):
        print(f"no experiments folder: {EXPERIMENTS}", file=sys.stderr)
        return 2
    return check([os.path.join(EXPERIMENTS, n) for n in experiment_folders(EXPERIMENTS)], whole=True, as_json=a.json)


if __name__ == "__main__":
    sys.exit(main())
