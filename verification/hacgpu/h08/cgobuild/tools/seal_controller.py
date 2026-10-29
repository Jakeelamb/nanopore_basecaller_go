#!/usr/bin/env python3
"""Seal only the maintained CGO0 controller; external supervision is mandatory."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from fractions import Fraction

DEADLINE = time.monotonic() + 175
GO_ROOT = Path('/usr/lib/go')
TARGET = './verification/hacgpu/h08/cgobuild/cmd'
SCHEMA = 'nanopore-h08-cgo-interposer/v1'
MODULE = 'module github.com/Jakeelamb/nanopore_basecaller_go'


def budget():
    left = DEADLINE - time.monotonic()
    if left <= 0:
        raise TimeoutError('shared 175-second deadline exceeded')
    return left


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            budget()
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f'canonical regular file required: {path}')
    return {'Path': str(path), 'SHA256': digest(path)}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
    Path(path).chmod(0o400)


def effective():
    rels = [s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines()
            if s.startswith('0::/')]
    if len(rels) != 1 or '..' in Path(rels[0]).parts:
        raise ValueError('bounded cgroup v2 required')
    base = Path('/sys/fs/cgroup')
    group = base / rels[0].lstrip('/')
    rows = []
    cpu, memory, swap, tasks = [], [], [], []
    for parent in [group, *group.parents]:
        if not parent.is_relative_to(base):
            break
        row = {'cgroup': str(parent)}
        for name, values in [('cpu.max', cpu), ('memory.max', memory),
                             ('memory.swap.max', swap), ('pids.max', tasks)]:
            p = parent / name
            if parent == base and not p.exists():
                continue
            text = p.read_text().strip()
            row[name] = text
            if name == 'cpu.max':
                quota, period = text.split()
                if quota != 'max':
                    values.append(Fraction(int(quota), int(period)))
            elif text != 'max':
                values.append(int(text))
        rows.append(row)
    if not all([cpu, memory, swap, tasks]) or (
            min(cpu), min(memory), min(swap), min(tasks)) != (2, 3 << 30, 0, 64):
        raise ValueError('effective 2CPU/3GiB/swap0/Tasks64 required')
    return rows


def authenticate(expected):
    for path, want in sorted(expected.items()):
        if pin(Path(path))['SHA256'] != want:
            raise ValueError(f'input changed before child: {path}')


def selected(path, root):
    text = Path(path).read_text()
    decoder, names = json.JSONDecoder(), set()
    while text.strip():
        obj, end = decoder.raw_decode(text.lstrip())
        text = text.lstrip()[end:]
        directory = Path(obj['Dir'])
        if (obj.get('Error') or not directory.is_absolute() or
                not (directory.is_relative_to(root) or directory.is_relative_to(GO_ROOT)) or
                any(obj.get(k) for k in ['CgoFiles', 'CFiles', 'CXXFiles'])):
            raise ValueError('invalid CGO0 source selection')
        for key in ['GoFiles', 'HFiles', 'SFiles', 'SysoFiles', 'EmbedFiles']:
            for name in obj.get(key, []):
                member = Path(name)
                if member.is_absolute() or '..' in member.parts:
                    raise ValueError('invalid selected source member')
                names.add(str(directory / member))
        mod = (obj.get('Module') or {}).get('GoMod')
        if mod:
            if Path(mod) != root / 'go.mod':
                raise ValueError('unexpected module selection')
            names.add(mod)
    if not names:
        raise ValueError('empty source selection')
    return {name: pin(Path(name))['SHA256'] for name in sorted(names)}


class Seal:
    def __init__(self, root, out, env, expected):
        self.root, self.out, self.env, self.expected = root, out, env, expected

    def run(self, label, argv, expected_exit=0):
        directory = self.out / 'commands' / label
        directory.mkdir()
        stdout, stderr = directory / 'stdout', directory / 'stderr'
        receipt = dict(command=argv, environment_overrides=[k + '=' + v for k, v in sorted(self.env.items())],
                       started_utc=utc(), exit_code=-1, stdout=str(stdout), stderr=str(stderr))
        write(directory / 'intent.json', dict(receipt, cwd=str(self.root)))
        process = {'PID': None, 'PGID': None}
        child, failure = None, None
        try:
            with stdout.open('xb') as a, stderr.open('xb') as b:
                authenticate(self.expected)
                effective()
                budget()
                child = subprocess.Popen(argv, cwd=self.root, env=self.env,
                                         stdin=subprocess.DEVNULL, stdout=a, stderr=b,
                                         start_new_session=True)
                process = {'PID': child.pid, 'PGID': child.pid}
                write(directory / 'process.json', process)
                while True:
                    remaining = budget()
                    if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                        break
                    time.sleep(min(0.01, remaining))
        except BaseException as exc:
            failure = exc
        finally:
            if child is not None:
                try:
                    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                except BaseException as exc:
                    failure = failure or exc
                else:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except BaseException as exc:
                        failure = failure or exc
                    try:
                        receipt['exit_code'] = child.wait(timeout=max(0.001, DEADLINE - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        failure = failure or TimeoutError('cleanup deadline exceeded')
                    except BaseException as exc:
                        failure = failure or exc
            receipt['finished_utc'] = utc()
            for key, path in [('stdout_sha256', stdout), ('stderr_sha256', stderr)]:
                with path.open('rb') as f:
                    receipt_digest = hashlib.file_digest(f, 'sha256')
                receipt[key] = receipt_digest.hexdigest()
            write(directory / 'command.json', receipt)
            write(directory / 'attempt.json', dict(process=process, error=str(failure) if failure else None))
        if failure:
            raise failure
        if receipt['exit_code'] != expected_exit:
            raise ValueError(f'{label}: exit {receipt["exit_code"]}, expected {expected_exit}')
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--tools-controller', required=True, type=Path)
    parser.add_argument('--tools-controller-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    out = None
    try:
        root = args.root
        if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError('canonical absolute checkout root required')
        anchors = [root / 'go.mod', root / TARGET[2:] / 'main.go']
        initial = {str(p): pin(p)['SHA256'] for p in anchors}
        if MODULE not in anchors[0].read_text().splitlines():
            raise ValueError('wrong source module')
        controller_pin = pin(args.tools_controller)
        if not re.fullmatch('[0-9a-f]{64}', args.tools_controller_sha256) or controller_pin['SHA256'] != args.tools_controller_sha256:
            raise ValueError('tools controller independent SHA256 differs')
        old_bytes = args.tools_controller.read_bytes()
        if hashlib.sha256(old_bytes).hexdigest() != args.tools_controller_sha256:
            raise ValueError('tools controller changed while reading')
        old = json.loads(old_bytes)
        tools = old['Tools']
        required = ['VERSION', 'go.env', 'bin/go', 'pkg/tool/linux_amd64/asm',
                    'pkg/tool/linux_amd64/compile', 'pkg/tool/linux_amd64/link',
                    'pkg/include/asm_amd64.h', 'pkg/include/funcdata.h', 'pkg/include/textflag.h']
        if not isinstance(tools, dict) or not all(str(GO_ROOT / p) in tools for p in required):
            raise ValueError('tools controller missing selected Go tools')
        for p, h in tools.items():
            if not Path(p).is_relative_to(GO_ROOT) or not re.fullmatch('[0-9a-f]{64}', h):
                raise ValueError('invalid authenticated Go tool pin')
        limits = effective()
        candidate = args.out
        base = root / 'artifacts/tensor'
        if (not candidate.is_absolute() or candidate == base or
                not candidate.is_relative_to(base) or candidate.resolve() != candidate):
            raise ValueError('fresh canonical out below root/artifacts/tensor required')
        if candidate.exists():
            raise ValueError('fresh out required; output reuse refused')
        authenticate(tools)
        if (GO_ROOT / 'VERSION').read_text().splitlines()[0] != 'go1.27.3':
            raise ValueError('authenticated Go1.27.3 required')
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.mkdir(mode=0o700)
        out = candidate
        for name in ['commands', 'sources', 'cache', 'tmp', 'gopath']:
            (out / name).mkdir(mode=0o700)
        write(out / 'effective-before.json', limits)
        write(out / 'tools-controller-pin.json', controller_pin)
        env = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', TZ='UTC',
                   GOROOT=str(GO_ROOT), GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off',
                   GOENV='off', GOWORK='off', GOFLAGS='', GOOS='linux', GOARCH='amd64',
                   GOAMD64='v1', CGO_ENABLED='0', GOMAXPROCS='2', GOMEMLIMIT='3GiB',
                   GOCACHE=str(out / 'cache'), GOTMPDIR=str(out / 'tmp'),
                   TMPDIR=str(out / 'tmp'), GOPATH=str(out / 'gopath'))
        seal = Seal(root, out, env, initial | tools | {controller_pin['Path']: controller_pin['SHA256']})
        go = str(GO_ROOT / 'bin/go')
        before = seal.run('list-before', [go, 'list', '-deps', '-json', TARGET])
        production = selected(before['stdout'], root)
        inputs = production | tools
        authenticate(seal.expected)
        seal.expected.update(inputs)
        write(out / 'inputs-before.json', inputs)
        snapshots = {}
        for path, want in sorted(inputs.items()):
            budget()
            dest = out / 'sources' / want
            body = Path(path).read_bytes()
            if hashlib.sha256(body).hexdigest() != want:
                raise ValueError(f'snapshot source changed: {path}')
            if not dest.exists():
                with dest.open('xb') as f:
                    f.write(body)
                dest.chmod(0o400)
            snapshots[path] = pin(dest)
            if snapshots[path]['SHA256'] != want:
                raise ValueError('snapshot differs')
        binary = out / 'cgo-plan'
        build = seal.run('build', [go, 'build', '-a', '-x', '-work', '-p=2', '-pgo=off',
                                  '-trimpath', '-buildvcs=false', '-o', str(binary), TARGET])
        binary.chmod(0o500)
        after = seal.run('list-after', [go, 'list', '-deps', '-json', TARGET])
        if selected(after['stdout'], root) != production:
            raise ValueError('selected source membership or bytes changed')
        authenticate(seal.expected)
        write(out / 'inputs-after.json', inputs)
        descriptor = dict(Schema=SCHEMA, Executable=pin(binary), Inputs=inputs,
                          Snapshots=snapshots, Tools=tools,
                          Lists=[pin(Path(r['stdout'])) for r in [before, after]], Build=build)
        descriptor_path = out / 'controller.json'
        write(descriptor_path, descriptor)
        seal.expected.update({str(binary): descriptor['Executable']['SHA256'],
                              str(descriptor_path): digest(descriptor_path)})
        for q in snapshots.values():
            seal.expected[q['Path']] = q['SHA256']
        def check(path):
            return [str(binary), '-mode', 'check-controller', '-controller', str(path),
                    '-controller-sha', digest(path)]
        seal.run('check-controller', check(descriptor_path))
        mutant = dict(descriptor, Snapshots=dict(snapshots))
        mutant['Snapshots'].pop(next(iter(mutant['Snapshots'])))
        mutant_path = out / 'missing-snapshot.json'
        write(mutant_path, mutant)
        rejection = seal.run('missing-snapshot', check(mutant_path), expected_exit=1)
        if 'custody incomplete' not in Path(rejection['stderr']).read_text():
            raise ValueError('missing snapshot did not produce custody-incomplete rejection')
        authenticate(seal.expected)
        write(out / 'effective-after.json', effective())
        write(out / 'summary.json', dict(Status='CGO0_SEALED', Controller=pin(descriptor_path),
                                         MissingSnapshot=pin(mutant_path), NativeCommands=0))
        return 0
    except Exception as exc:
        if out is not None:
            write(out / 'failure.json', dict(Error=str(exc), UTC=utc()))
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
