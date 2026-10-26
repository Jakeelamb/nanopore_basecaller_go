import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import selectors
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-20261025')
OUT = ROOT / 'artifacts/hac-vcs-aux/host-003'
SCRIPT = ROOT / 'verification/hacvcsaux/host3.py'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-host-20261025-003.service'
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC',
       'GOROOT': '/usr/lib/go', 'GOTOOLCHAIN': 'local', 'GOPROXY': 'off',
       'GOSUMDB': 'off', 'GOENV': 'off', 'GOWORK': 'off', 'GOFLAGS': '',
       'GOOS': 'linux', 'GOARCH': 'amd64', 'GOAMD64': 'v1', 'CGO_ENABLED': '0',
       'GOMAXPROCS': '2', 'GOMEMLIMIT': '3GiB', 'GOCACHE': str(OUT / 'gocache'),
       'GOTMPDIR': str(OUT / 'tmp'), 'TMPDIR': str(OUT / 'tmp'), 'GOPATH': str(OUT / 'gopath'),
       'UV_PYTHON': '/usr/bin/python3.14', 'UV_PYTHON_DOWNLOADS': 'never', 'UV_OFFLINE': '1',
       'UV_CACHE_DIR': str(OUT / 'uvcache')}
TOOLS = [pathlib.Path('/usr/lib/go/bin/go'), pathlib.Path('/usr/lib/go/bin/gofmt')] + [
    pathlib.Path('/usr/lib/go/pkg/tool/linux_amd64') / name for name in
    ['vet', 'preprofile', 'link', 'fix', 'cover', 'compile', 'cgo', 'asm']]
TOOLS += [pathlib.Path('/usr/lib/go/go.env'), pathlib.Path('/usr/lib/go/VERSION')]
TOOLS += sorted(path for path in pathlib.Path('/usr/lib/go/pkg/include').rglob('*') if path.is_file())
TOOLS += [pathlib.Path('/usr/bin/python3.14'), pathlib.Path('/home/jake/.local/bin/uv')]
FORMAT = ['internal/gpubatch/aux.go', 'internal/gpubatch/lstm_packing.go', 'internal/gpubatch/lstm_packing_test.go',
          'verification/hacvcsaux/fixtures.go', 'verification/hacvcsaux/fixtures_test.go',
          'verification/hacvcsaux/inputs.go', 'verification/hacvcsaux/compiler.go',
          'verification/hacvcsaux/compiler_test.go', 'verification/hacvcsaux/compare.go',
          'verification/hacvcsaux/cmd/main.go']
COMMANDS = [
    ('syntax', [sys.executable, '-c', 'import ast,pathlib; paths=sorted(pathlib.Path("verification/hacvcsaux").glob("*.py")); [ast.parse(path.read_text(), filename=str(path)) for path in paths]; print("parsed",len(paths),"fixed verification scripts")'], 20),
    ('format', ['/usr/lib/go/bin/gofmt', '-w'] + FORMAT, 20),
    ('selection', ['/usr/lib/go/bin/go', 'list', '-deps', '-test', '-json', './internal/gpubatch', './verification/hacvcsaux/...'], 30),
    ('tests', ['/usr/lib/go/bin/go', 'test', '-json', '-count=1', '-p=2', '-pgo=off', './internal/gpubatch', './verification/hacvcsaux/...'], 130),
    ('vet', ['/usr/lib/go/bin/go', 'vet', '-p=2', '-pgo=off', './internal/gpubatch', './verification/hacvcsaux/...'], 40),
    ('inputs', ['/usr/lib/go/bin/go', 'run', '-p=2', '-pgo=off', './verification/hacvcsaux/cmd', '-unit', 'inputs', '-out', str(OUT / 'native-inputs.txt')], 40),
    ('seal', ['/usr/lib/go/bin/go', 'build', '-p=2', '-pgo=off', '-trimpath', '-o', str(OUT / 'hacvcsaux'), './verification/hacvcsaux/cmd'], 60),
    ('compiler-reader', [str(OUT / 'hacvcsaux'), '-unit', 'compiler-check'], 30),
    ('compiler-held', [sys.executable, str(ROOT / 'verification/hacvcsaux/compile.py')], 20),
]

def sources():
    paths = [path for path in ROOT.rglob('*.go') if not any(part in ['artifacts', 'external', '.git'] for part in path.relative_to(ROOT).parts)]
    paths += [ROOT / 'go.mod', ROOT / 'verification/hacvcsaux/fixtures.json', SCRIPT,
              ROOT / 'verification/hacvcsaux/helpers-baseline.json', ROOT / 'verification/hacvcsaux/oracle-plan.json']
    paths += sorted((ROOT / 'verification/hacvcsaux').glob('*.py'))
    paths += sorted((ROOT / 'verification/hacvcsaux/native').glob('*'))
    return {str(path): sha(path) for path in sorted(paths)}

def selection(path):
    decoder = json.JSONDecoder()
    body = path.read_text()
    pins, packages, generated = {}, [], []
    while body.strip():
        package, end = decoder.raw_decode(body.lstrip())
        body = body.lstrip()[end:]
        assert not package.get('Error') and not package.get('DepsErrors'), package
        directory = pathlib.Path(package.get('Dir', ''))
        assert directory.is_absolute(), package
        assert not any(package.get(name) for name in ['CgoFiles', 'CFiles', 'CXXFiles', 'MFiles', 'FFiles', 'SwigFiles', 'SwigCXXFiles']), package
        packages.append(package)
        for family in ['GoFiles', 'HFiles', 'SFiles', 'SysoFiles', 'EmbedFiles']:
            for name in package.get(family, []):
                member = pathlib.Path(name)
                if not member.is_absolute():
                    assert '..' not in member.parts
                    member = directory / member
                if not member.exists():
                    assert package.get('Name') == 'main' and package.get('ImportPath', '').endswith('.test') and family == 'GoFiles' and member.name == '_testmain.go', (package.get('ImportPath'), family, name)
                    generated.append({'package': package['ImportPath'], 'family': family, 'path': str(member), 'producer': '/usr/lib/go/bin/go'})
                    continue
                assert member.is_file()
                pins[str(member.resolve())] = sha(member)
        module = package.get('Module')
        if module and module.get('GoMod'):
            member = pathlib.Path(module['GoMod']).resolve()
            pins[str(member)] = sha(member)
    assert pins and packages
    result = {'schema': 'hac-vcs-aux-go-selection/v1', 'inputs': pins, 'packages': packages, 'generated_test_main': generated}
    write('selection.json', result)
    return pins

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def write(name, value):
    with (OUT / name).open('x') as target:
        json.dump(value, target, indent=2)
        target.write('\n')

def effective():
    rel = next(line[3:] for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::/'))
    group = pathlib.Path('/sys/fs/cgroup') / rel.lstrip('/')
    result = {'cgroup': str(group), 'pid': os.getpid(), 'parent_pid': os.getppid()}
    for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']:
        result[name] = (group / name).read_text().strip()
    assert [result[name] for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']] == ['200000 100000', '3221225472', '0', '64']
    return result

def run(label, argv, stdout, cap, seconds):
    receipt = {'argv': argv, 'environment': ENV, 'started_utc': utc(), 'stdout_limit': cap, 'stderr_limit': 65536, 'process_seconds': seconds}
    child = subprocess.Popen(argv, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    receipt['pid'] = child.pid
    deadline = time.monotonic() + seconds
    streams = selectors.DefaultSelector()
    streams.register(child.stdout, selectors.EVENT_READ, 'stdout')
    streams.register(child.stderr, selectors.EVENT_READ, 'stderr')
    counts = {'stdout': 0, 'stderr': 0}
    digest = {name: hashlib.sha256() for name in counts}
    failure = None
    stderr = OUT / (label + '.stderr')
    with stdout.open('xb') as output, stderr.open('xb') as errors:
        destinations = {'stdout': output, 'stderr': errors}
        caps = {'stdout': cap, 'stderr': 65536}
        while streams.get_map():
            if time.monotonic() > deadline and failure is None:
                failure = 'native_process_timeout'
                os.killpg(child.pid, signal.SIGKILL)
            for key, _ in streams.select(timeout=0.05):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    streams.unregister(key.fileobj)
                    continue
                name = key.data
                counts[name] += len(block)
                digest[name].update(block)
                target = destinations[name]
                target.write(block[:max(0, caps[name] - target.tell())])
                if counts[name] > caps[name] and failure is None:
                    failure = name + '_limit_exceeded'
                    os.killpg(child.pid, signal.SIGKILL)
        receipt['exit'] = child.wait()
    streams.close()
    receipt.update({'finished_utc': utc(), 'stream_bytes': counts, 'stream_sha256': {name: value.hexdigest() for name, value in digest.items()}, 'failure': failure, 'stdout': {'path': str(stdout), 'sha256': sha(stdout)}, 'stderr': {'path': str(stderr), 'sha256': sha(stderr)}})
    write(label + '-receipt.json', receipt)
    expected_exit = 75 if label == 'compiler-held' else 0
    assert receipt['exit'] == expected_exit and failure is None
    return receipt

def inner():
    receipt = {'started_utc': utc(), 'effective_before': effective(), 'environment': ENV, 'commands': []}
    assert pathlib.Path(sys.executable).resolve() == pathlib.Path('/usr/bin/python3.14')
    receipt['python_executable'] = sys.executable
    descriptor = os.open(LOCK, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        receipt.update({'status': 'deferred_host_lock', 'exit': 75, 'finished_utc': utc()})
        write('receipt.json', receipt)
        return 75
    receipt['lock_acquired_utc'] = utc()
    code = 1
    try:
        receipt['tools_before'] = {str(path): sha(path) for path in TOOLS}
        plan = json.loads((ROOT / 'verification/hacvcsaux/oracle-plan.json').read_text())
        receipt['oracle_inputs_before'] = {path: sha(path) for path in plan['InitialInputs']}
        assert receipt['oracle_inputs_before'] == plan['InitialInputs']
        receipt['source_before_format'] = sources()
        assert receipt['source_before_format'][str(ROOT / 'verification/hacvcsaux/fixtures.json')] == 'd4602d3fd2ac2e45515a949087b82fe50fe5a0a92e3dcffc66b82979d0c9e906'
        helper_baseline = json.loads((ROOT / 'verification/hacvcsaux/helpers-baseline.json').read_text())
        for name, expected in helper_baseline['files'].items():
            assert sha(ROOT / name) == expected == sha(pathlib.Path(helper_baseline['canonical_root']) / name)
        for label, argv, seconds in COMMANDS:
            receipt['commands'].append(run(label, argv, OUT / (label + '.stdout'), 2 * 1024 * 1024, seconds))
            current = sources()
            if label == 'format':
                receipt['source_after_format'] = current
                changed = [path for path in current if current[path] != receipt['source_before_format'].get(path)]
                assert all(path in [str(ROOT / name) for name in FORMAT] for path in changed)
            elif label != 'syntax':
                assert current == receipt['source_after_format']
            if label == 'seal':
                receipt['checker_sha256'] = sha(OUT / 'hacvcsaux')
            if label == 'compiler-reader':
                compiler = json.loads((OUT / 'compiler-reader.stdout').read_text())
                assert compiler['Schema'] == 'hac-vcs-aux-compiler-evidence/v1' and not compiler['ExecutionAdmitted']
                receipt['compiler_input_pins'] = compiler['Compiler']['Inputs']
            if label == 'compiler-held':
                held = json.loads((OUT / 'compiler-held.stdout').read_text())
                assert held == {'Status': 'compiler_held', 'exit': 75}
                assert not (ROOT / 'artifacts/hac-vcs-aux/build-001').exists()
            if label == 'selection':
                receipt['selected_inputs'] = selection(OUT / 'selection.stdout')
            if 'selected_inputs' in receipt:
                assert {path: sha(path) for path in receipt['selected_inputs']} == receipt['selected_inputs']
            assert {str(path): sha(path) for path in TOOLS} == receipt['tools_before']
            assert {path: sha(path) for path in receipt['oracle_inputs_before']} == receipt['oracle_inputs_before']
        receipt['native_inputs_sha256'] = sha(OUT / 'native-inputs.txt')
        assert receipt['native_inputs_sha256'] == '5532c32a6b6ff13238e563a0ce4ec873a7018300f70abf5848195210c77dcf4d'
        receipt['effective_after'] = effective()
        code = 0
    except BaseException as error:
        receipt['error'] = repr(error)
    finally:
        try:
            receipt['source_after'] = sources()
            receipt['tools_after'] = {str(path): sha(path) for path in TOOLS}
            receipt['oracle_inputs_after'] = {path: sha(path) for path in receipt['oracle_inputs_before']}
            assert receipt['oracle_inputs_after'] == receipt['oracle_inputs_before']
            receipt['effective_after'] = effective()
            assert receipt['tools_before'] == receipt['tools_after']
            if 'selected_inputs' in receipt:
                receipt['selected_inputs_after'] = {path: sha(path) for path in receipt['selected_inputs']}
                assert receipt['selected_inputs'] == receipt['selected_inputs_after']
        except BaseException as error:
            code = 1
            receipt['after_check_error'] = repr(error)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        receipt.update({'lock_released_utc': utc(), 'finished_utc': utc(), 'exit': code})
        write('receipt.json', receipt)
    return code

if '--inner' in sys.argv:
    raise SystemExit(inner())

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir()
for directory in ['tmp', 'gocache', 'gopath', 'uvcache']:
    (OUT / directory).mkdir()
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT, '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=3G', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=175s', '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s',
        '--setenv=UV_PYTHON_DOWNLOADS=never', '--setenv=UV_OFFLINE=1', '--setenv=UV_PYTHON=/usr/bin/python3.14', '--setenv=UV_CACHE_DIR=' + str(OUT / 'uvcache'),
        '/home/jake/.local/bin/uv', 'run', '--no-project', '--offline', '--python', '/usr/bin/python3.14', 'python', str(SCRIPT), '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(), 'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'), 'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'receipt.json')}))
raise SystemExit(result.returncode)
