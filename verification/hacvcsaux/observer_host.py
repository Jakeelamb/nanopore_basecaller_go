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

ROOT = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-observer-20261025')
OUT = ROOT / 'artifacts/hac-vcs-aux-observer/host-001'
SCRIPT = ROOT / 'verification/hacvcsaux/observer_host.py'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-observer-host-20261025-001.service'
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
FORMAT = ['verification/hacvcsaux/compare.go', 'verification/hacvcsaux/maps_test.go']
EVIDENCE = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-20261025/artifacts/hac-vcs-aux/first-001/data/evidence.json')
EVIDENCE_SHA = '2d22bad2ef879aaeb1eb72c90be0f8b317b037e89dd6837653b0c0c0967c886b'
COMMANDS = [
    ('format', ['/usr/lib/go/bin/gofmt', '-w'] + FORMAT, 20),
    ('selection', ['/usr/lib/go/bin/go', 'list', '-deps', '-test', '-json', './verification/hacvcsaux/...'], 30),
    ('tests', ['/usr/lib/go/bin/go', 'test', '-json', '-count=1', '-p=2', '-pgo=off', '-run', '^(TestMaps|TestIndependentLoaderCache)', './verification/hacvcsaux'], 130),
    ('vet', ['/usr/lib/go/bin/go', 'vet', '-p=2', '-pgo=off', './verification/hacvcsaux/...'], 40),
    ('seal', ['/usr/lib/go/bin/go', 'build', '-p=2', '-pgo=off', '-trimpath', '-o', str(OUT / 'hacvcsaux-observer'), './verification/hacvcsaux/cmd'], 60),
    ('readback', [str(OUT / 'hacvcsaux-observer'), '-unit', 'compare', '-evidence', str(EVIDENCE), '-evidence-sha', EVIDENCE_SHA], 60),
]


def sources():
    paths = [p for p in ROOT.rglob('*.go') if not any(x in ['artifacts', 'external', '.git'] for x in p.relative_to(ROOT).parts)]
    paths += [ROOT / 'go.mod', ROOT / 'verification/hacvcsaux/fixtures.json', SCRIPT,
              ROOT / 'verification/hacvcsaux/helpers-baseline.json', ROOT / 'verification/hacvcsaux/observer-baseline.json']
    paths += sorted((ROOT / 'verification/hacvcsaux/native').glob('*'))
    return {str(p): sha(p) for p in sorted(paths)}


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
    receipt = {'Schema': 'hac-vcs-aux-observer-host/v1', 'Status': 'failed', 'started_utc': utc(),
               'effective_before': effective(), 'environment': ENV, 'commands': [],
               'NativeRerunPermitted': False, 'NativeProducerExecutedThisTurn': False}
    descriptor = os.open(LOCK, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        receipt.update({'Status': 'deferred_host_lock', 'exit': 75, 'finished_utc': utc()})
        write('receipt.json', receipt)
        return 75
    receipt['lock_acquired_utc'] = utc()
    code = 1
    producer_inputs, captures, bindings = {}, {}, {}
    try:
        baseline_path = ROOT / 'verification/hacvcsaux/observer-baseline.json'
        assert sha(baseline_path) == '8e3b42f9f43eebad580e56a00ecab04a27dba5c345f2e4f6b7da59c2dce46765'
        baseline = json.loads(baseline_path.read_text())
        assert baseline['Schema'] == 'hac-vcs-aux-observer-baseline/v1'
        assert baseline['ConsumerRoot'] == str(ROOT) and not baseline['NativeRerunPermitted']
        producer = pathlib.Path(baseline['ProducerRoot'])
        for name in ['ProducerBuild', 'NativeEvidence', 'ProducerFailedRuntime', 'ProducerCleanup']:
            selected = baseline[name]
            assert sha(selected['Path']) == selected['SHA256']
            bindings[selected['Path']] = selected['SHA256']
        build = json.loads(pathlib.Path(baseline['ProducerBuild']['Path']).read_text())
        runtime = json.loads(pathlib.Path(baseline['ProducerFailedRuntime']['Path']).read_text())
        cleanup = json.loads(pathlib.Path(baseline['ProducerCleanup']['Path']).read_text())
        evidence = json.loads(EVIDENCE.read_text())
        assert str(EVIDENCE) == baseline['NativeEvidence']['Path'] and sha(EVIDENCE) == EVIDENCE_SHA
        assert build['Status'] == 'built_outputs_unexecuted' and build['exit'] == 0
        assert runtime['Status'] == 'failed' and runtime['exit'] == 1 and runtime['oracle_child_started']
        assert runtime['before'] == runtime['after'] and len(runtime['commands']) == 3
        assert [c['exit'] for c in runtime['commands']] == [0, 0, 1]
        assert all(c['failure'] is None for c in runtime['commands'])
        native = runtime['commands'][1]
        assert native['argv'] == [build['Output']['Path'], evidence['Inputs']['Path'], str(EVIDENCE.parent), 'first']
        assert cleanup['cgroup_absent'] and cleanup['before_after_inputs_equal']
        assert all(x['original_birth_absent'] for x in cleanup['birth_checks'])
        assert all(x['absent'] for x in cleanup['child_process_groups'])
        assert native['pid'] in cleanup['expected_runtime_pids_birth_observed']
        assert runtime['Evidence'] == baseline['NativeEvidence']
        assert evidence['Oracle'] == build['Output'] and evidence['Build'] == baseline['ProducerBuild']
        assert evidence['Selection'] == 'first' and evidence['Status'] == 'captured_public_host_cpu'
        original_error = pathlib.Path(baseline['ProducerFailedRuntime']['Path']).parent / 'compare.stderr'
        assert sha(original_error) == runtime['commands'][2]['stream_sha256']['stderr']
        assert original_error.read_text().strip() == 'unapproved native mapping: /etc/ld.so.cache'
        bindings[str(original_error)] = sha(original_error)
        producer_inputs.update(build['Inputs'])
        producer_inputs.update(build['Objects'])
        producer_inputs[build['Output']['Path']] = build['Output']['SHA256']
        captures.update({str(EVIDENCE.parent / name): value for name, value in evidence['Files'].items()})
        assert len(evidence['Files']) == 71
        producer_host_path = producer / 'artifacts/hac-vcs-aux/host-003/receipt.json'
        assert sha(producer_host_path) == build['Inputs'][str(producer_host_path)]
        producer_host = json.loads(producer_host_path.read_text())
        receipt['tools_before'] = {str(p): sha(p) for p in TOOLS}
        assert receipt['tools_before'] == producer_host['tools_before']
        for name, expected in baseline['CopiedSourcePins'].items():
            if name != 'verification/hacvcsaux/compare.go':
                assert sha(ROOT / name) == expected == sha(producer / name), 'copied source changed: ' + name
        for name, original in build['SourcePins'].items():
            assert sha(ROOT / 'verification/hacvcsaux/native' / name) == original['SHA256'] == sha(original['Path'])
        receipt['source_before_format'] = sources()
        receipt['ProducerBinding'] = {'Build': baseline['ProducerBuild'], 'Evidence': baseline['NativeEvidence'],
                                      'FailedRuntime': baseline['ProducerFailedRuntime'], 'Cleanup': baseline['ProducerCleanup'],
                                      'OriginalNativeCommand': native, 'OriginalMappedCache': '/etc/ld.so.cache'}

        def unchanged():
            for family in [producer_inputs, captures, bindings]:
                for path, expected in family.items():
                    assert sha(path) == expected, 'immutable producer/capture changed: ' + path
            assert {str(p): sha(p) for p in TOOLS} == receipt['tools_before']

        unchanged()
        receipt['producer_input_pins_before'] = dict(producer_inputs)
        receipt['capture_pins_before'] = dict(captures)
        for label, argv, seconds in COMMANDS:
            receipt['commands'].append(run(label, argv, OUT / (label + '.stdout'), 2 * 1024 * 1024, seconds))
            current = sources()
            if label == 'format':
                receipt['source_after_format'] = current
                changed = [path for path in current if current[path] != receipt['source_before_format'].get(path)]
                assert all(path in [str(ROOT / name) for name in FORMAT] for path in changed)
            else:
                assert current == receipt['source_after_format']
            if label == 'selection':
                receipt['selected_inputs'] = selection(OUT / 'selection.stdout')
            if 'selected_inputs' in receipt:
                assert {path: sha(path) for path in receipt['selected_inputs']} == receipt['selected_inputs']
            if label == 'seal':
                receipt['checker_sha256'] = sha(OUT / 'hacvcsaux-observer')
            if label == 'readback':
                comparison = json.loads((OUT / 'readback.stdout').read_text())
                assert comparison['Schema'] == 'hac-vcs-aux-comparison/v1' and comparison['Cases'] == 1 and comparison['Runs'] == 4
                assert comparison['OutputWordRejection'].startswith('InLayout[0]')
                receipt['Comparison'] = comparison
            unchanged()
        receipt['Status'] = 'retained_first_numerical_observation_passed'
        code = 0
    except BaseException as error:
        receipt['error'] = repr(error)
    finally:
        try:
            receipt['source_after'] = sources()
            receipt['tools_after'] = {str(p): sha(p) for p in TOOLS}
            receipt['producer_input_pins_after'] = {path: sha(path) for path in producer_inputs}
            receipt['capture_pins_after'] = {path: sha(path) for path in captures}
            receipt['bindings_after'] = {path: sha(path) for path in bindings}
            receipt['effective_after'] = effective()
            assert producer_inputs == receipt['producer_input_pins_after'] and captures == receipt['capture_pins_after']
            assert bindings == receipt['bindings_after']
            if 'tools_before' in receipt:
                assert receipt['tools_before'] == receipt['tools_after']
            if 'source_after_format' in receipt:
                assert receipt['source_after'] == receipt['source_after_format']
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
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT,
        '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=3G',
        '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=175s',
        '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s',
        '--setenv=UV_PYTHON_DOWNLOADS=never', '--setenv=UV_OFFLINE=1', '--setenv=UV_PYTHON=/usr/bin/python3.14', '--setenv=UV_CACHE_DIR=' + str(OUT / 'uvcache'),
        '/home/jake/.local/bin/uv', 'run', '--no-project', '--offline', '--python', '/usr/bin/python3.14', 'python', str(SCRIPT), '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(),
                            'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'),
                            'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'receipt.json')}))
raise SystemExit(result.returncode)
