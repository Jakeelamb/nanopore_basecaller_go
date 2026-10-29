#!/usr/bin/env python3
"""Compile four unexecuted metadata boundary fixtures inside an owned unit.

The controller supplies the stock lock, cgroup caps and watchdog before startup.
This script never runs the resulting CLI or providers. Its raw traces supplement
the pinned source/compiler and output receipts; they are not a closed-build proof.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"Path": str(path), "SHA256": digest, "Bytes": path.stat().st_size}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = Path(args.out)
    if not out.is_absolute() or out != out.resolve() or not out.is_relative_to(root / "artifacts"):
        raise ValueError("fresh canonical output under this checkout's artifacts required")
    compiler = pin(args.compiler)
    if compiler["Path"] != args.compiler or compiler["SHA256"] != args.compiler_sha256:
        raise ValueError("selected compiler differs")
    sources = root / "verification/supgpu/graph/testdata"
    source_pins = [pin(sources / name) for name in ("metadata-cli.c", "metadata-provider.c")]
    tracer = pin("/usr/bin/strace")
    out.mkdir(parents=True)
    temporary = out / "tmp"
    temporary.mkdir()
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "TMPDIR": str(temporary)}
    markers = {letter: out / ("marker-" + letter) for letter in ("a", "b")}
    records = []
    outputs = {}
    save(out / "intent.json", {"started": now(), "compiler": compiler, "sources": source_pins,
                              "tracer": tracer, "environment": environment, "script": pin(__file__)})
    for name in ("provider-a", "provider-b", "matched", "mismatched"):
        directory = out / name
        directory.mkdir()
        provider = name.startswith("provider-")
        letter = name[-1] if provider else ("a" if name == "matched" else "b")
        binary = directory / ("libmetadata-provider.so" if provider else "nvidia-smi")
        source = sources / ("metadata-provider.c" if provider else "metadata-cli.c")
        argv = [args.compiler, "-v", "-MD", "-MF", str(directory / "dependencies.d")]
        if provider:
            argv += ["-shared", "-fPIC", "-Wl,-soname,libmetadata-provider.so",
                     "-DCONSTRUCTOR_MARKER=" + json.dumps(str(markers[letter]))]
        else:
            library = out / ("provider-" + letter) / "libmetadata-provider.so"
            argv += ["-DSELECT_PROVIDER_PATH=" + json.dumps(str(library))]
        argv += [str(source), "-Wl,-Map," + str(directory / "link.map"), "-o", str(binary)]
        if not provider:
            argv += ["-ldl"]
        command = [tracer["Path"], "-f", "-yy", "-s", "4096", "-o", str(directory / "trace"),
                   "-e", "trace=execve,openat,newfstatat,statx,readlink", "--", *argv]
        record = {"argv": argv, "trace_argv": command, "started": now()}
        save(directory / "intent.json", record)
        try:
            with (directory / "stdout").open("xb") as stdout, (directory / "stderr").open("xb") as stderr:
                record["exit_code"] = subprocess.run(command, cwd=root, env=environment,
                                                     stdout=stdout, stderr=stderr, timeout=30).returncode
        except subprocess.TimeoutExpired:
            record["exit_code"] = 124
        finally:
            record["finished"] = now()
            record["retained_files"] = [pin(p) for p in sorted(directory.iterdir()) if p.is_file()]
            save(directory / "receipt.json", record)
        records.append(record)
        if record["exit_code"] != 0:
            raise RuntimeError("fixture compiler failed; retained raw receipt: " + name)
        outputs[name] = pin(binary)
    if compiler != pin(args.compiler) or source_pins != [pin(p["Path"]) for p in source_pins]:
        raise ValueError("source/compiler changed during compilation")
    if any(p.exists() for p in markers.values()):
        raise ValueError("unexecuted fixture unexpectedly has constructor markers")
    result = {"schema": "nanopore-sup-metadata-fixture-build/v1", "status": "compiled_unexecuted",
              "finished": now(), "compiler": compiler, "sources": source_pins, "outputs": outputs,
              "markers": {k: str(v) for k, v in markers.items()}, "commands": records}
    save(out / "build.json", result)
    print(json.dumps(pin(out / "build.json")))


if __name__ == "__main__":
    main()
