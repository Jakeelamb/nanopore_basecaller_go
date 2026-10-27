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
SCRIPT = ROOT / 'verification/hacvcsaux/matrix.py'
BUILD = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-20261025/artifacts/hac-vcs-aux/build-004/build.json')
CHECKER = ROOT / 'artifacts/hac-vcs-aux-observer/host-001/hacvcsaux-observer'
INPUTS = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-20261025/artifacts/hac-vcs-aux/host-003/native-inputs.txt')
INPUTS_SHA = '5532c32a6b6ff13238e563a0ce4ec873a7018300f70abf5848195210c77dcf4d'
OBSERVER = ROOT / 'artifacts/hac-vcs-aux-observer/host-001/receipt.json'
OBSERVER_SHA = '70fb53e4a5370400a314ee12dc542eac78669e42c07881cbb4dead35bb6f760d'
CHECKER_SHA = 'dfaa28f18ba25975267f8d7a2cbcd39dffa74081ff153fe58e7c8fce36b4a7aa'
MATRIX_SHA = 'd4602d3fd2ac2e45515a949087b82fe50fe5a0a92e3dcffc66b82979d0c9e906'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(root, name, value):
    with (root / name).open('x') as target:
        json.dump(value, target, indent=2)
        target.write('\n')


def effective():
    relative = next(line[3:] for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::/'))
    group = pathlib.Path('/sys/fs/cgroup') / relative.lstrip('/')
    result = {'cgroup': str(group), 'pid': os.getpid(), 'parent_pid': os.getppid()}
    for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']:
        result[name] = (group / name).read_text().strip()
    assert [result[name] for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']] == ['200000 100000', '3221225472', '0', '64']
    return result


def execute(root, environment, label, argv, seconds):
    receipt = {'argv': argv, 'environment': environment, 'started_utc': utc(),
               'process_seconds': seconds, 'stdout_limit': 2 * 1024 * 1024, 'stderr_limit': 65536}
    child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    receipt['pid'] = child.pid
    deadline = time.monotonic() + seconds
    streams = selectors.DefaultSelector()
    streams.register(child.stdout, selectors.EVENT_READ, 'stdout')
    streams.register(child.stderr, selectors.EVENT_READ, 'stderr')
    counts = {'stdout': 0, 'stderr': 0}
    hashes = {name: hashlib.sha256() for name in counts}
    failure = None
    with (root / (label + '.stdout')).open('xb') as output, (root / (label + '.stderr')).open('xb') as errors:
        destinations = {'stdout': output, 'stderr': errors}
        limits = {'stdout': 2 * 1024 * 1024, 'stderr': 65536}
        while streams.get_map():
            if time.monotonic() > deadline and failure is None:
                failure = 'process_timeout'
                os.killpg(child.pid, signal.SIGKILL)
            for key, _ in streams.select(timeout=0.05):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    streams.unregister(key.fileobj)
                    continue
                name = key.data
                counts[name] += len(block)
                hashes[name].update(block)
                target = destinations[name]
                target.write(block[:max(0, limits[name] - target.tell())])
                if counts[name] > limits[name] and failure is None:
                    failure = name + '_limit_exceeded'
                    os.killpg(child.pid, signal.SIGKILL)
        receipt['exit'] = child.wait()
    streams.close()
    receipt.update({'finished_utc': utc(), 'failure': failure, 'stream_bytes': counts,
                    'stream_sha256': {name: value.hexdigest() for name, value in hashes.items()}})
    write(root, label + '-receipt.json', receipt)
    return receipt


def inner(root, selection, build_sha, input_sha):
    environment = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC',
                   'TMPDIR': str(root / 'tmp'), 'LD_LIBRARY_PATH': '/usr/lib',
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'GOMAXPROCS': '2', 'GOMEMLIMIT': '3GiB',
                   'UV_PYTHON': '/usr/bin/python3.14', 'UV_PYTHON_DOWNLOADS': 'never', 'UV_OFFLINE': '1',
                   'UV_CACHE_DIR': str(root / 'uvcache')}
    receipt = {'Schema': 'hac-vcs-aux-runtime/v1', 'Status': 'failed', 'started_utc': utc(),
               'effective_before': effective(), 'environment': environment, 'commands': [],
               'oracle_child_started': False, 'GPUQualified': False, 'Selection': selection}
    descriptor = os.open(LOCK, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        receipt.update({'Status': 'deferred_host_lock', 'exit': 75, 'finished_utc': utc()})
        write(root, 'runtime.json', receipt)
        return 75
    receipt['lock_acquired_utc'] = utc()
    code = 1
    pins = {}
    try:
        if input_sha != INPUTS_SHA or sha(INPUTS) != INPUTS_SHA:
            receipt['Status'] = 'rejected_input_pin_before_child'
            code = 75
            return code
        assert sha(BUILD) == build_sha
        build = json.loads(BUILD.read_text())
        assert build['Schema'] == 'hac-vcs-aux-build/v1' and build['Status'] == 'built_outputs_unexecuted' and build['exit'] == 0
        assert sha(OBSERVER) == OBSERVER_SHA
        observation = json.loads(OBSERVER.read_text())
        assert observation['Schema'] == 'hac-vcs-aux-observer-host/v1' and observation['Status'] == 'retained_first_numerical_observation_passed' and observation['exit'] == 0
        assert observation['source_after_format'] == observation['source_after']
        assert observation['producer_input_pins_before'] == observation['producer_input_pins_after']
        assert observation['capture_pins_before'] == observation['capture_pins_after']
        assert not observation['NativeProducerExecutedThisTurn']
        assert sha(CHECKER) == observation['checker_sha256'] == CHECKER_SHA
        for family in [build['Inputs'], build['Objects'], observation['selected_inputs'], observation['source_after'], observation['tools_before']]:
            for path, expected in family.items():
                assert path not in pins or pins[path] == expected, 'conflicting producer/consumer pin: ' + path
                pins[path] = expected
        pins.update({str(OBSERVER): OBSERVER_SHA, str(CHECKER): CHECKER_SHA})
        oracle = build['Output']
        pins[oracle['Path']] = oracle['SHA256']
        pins.update({str(BUILD): build_sha, str(INPUTS): INPUTS_SHA, str(SCRIPT): sha(SCRIPT)})
        assert str(CHECKER) in pins

        def check():
            for path, expected in pins.items():
                assert sha(path) == expected, 'runtime input changed: ' + path

        check()
        receipt['before'] = dict(pins)
        result = execute(root, environment, 'elf-before', [str(CHECKER), '-unit', 'artifact-check', '-oracle', oracle['Path'], '-oracle-sha', oracle['SHA256']], 30)
        receipt['commands'].append(result)
        assert result['exit'] == 0 and result['failure'] is None
        check()
        target = root / 'data'
        argv = [oracle['Path'], str(INPUTS), str(target), selection]
        receipt['oracle_child_started'] = True
        result = execute(root, environment, 'public-host', argv, 120)
        receipt['commands'].append(result)
        check()
        assert result['exit'] == 0 and result['failure'] is None
        files = {}
        for path in sorted(target.rglob('*')):
            assert not path.is_symlink()
            if path.is_file():
                files[str(path.relative_to(target))] = sha(path)
        evidence = {'Schema': 'hac-vcs-aux-native-evidence/v1', 'Status': 'captured_public_host_cpu',
                    'Selection': selection, 'MatrixSHA256': MATRIX_SHA, 'InputsSHA256': INPUTS_SHA,
                    'CompilerBaselineSHA256': build['CompilerBaselineSHA256'], 'Oracle': oracle,
                    'Build': {'Path': str(BUILD), 'SHA256': build_sha},
                    'Inputs': {'Path': str(INPUTS), 'SHA256': INPUTS_SHA}, 'Files': files,
                    'CorrectedObservation': {'Path': str(OBSERVER), 'SHA256': OBSERVER_SHA},
                    'GoVerifier': {'Path': str(CHECKER), 'SHA256': CHECKER_SHA}}
        write(target, 'evidence.json', evidence)
        evidence_path = target / 'evidence.json'
        receipt['Evidence'] = {'Path': str(evidence_path), 'SHA256': sha(evidence_path)}
        result = execute(root, environment, 'compare', [str(CHECKER), '-unit', 'compare', '-evidence', str(evidence_path), '-evidence-sha', sha(evidence_path)], 40)
        receipt['commands'].append(result)
        assert result['exit'] == 0 and result['failure'] is None
        comparison = json.loads((root / 'compare.stdout').read_text())
        assert comparison['Schema'] == 'hac-vcs-aux-comparison/v1' and comparison['OutputWordRejection'].startswith('InLayout[0]')
        receipt['ComparisonSHA256'] = sha(root / 'compare.stdout')
        receipt['Status'] = 'public_cpu_oracle_differential_passed'
        code = 0
    except BaseException as error:
        receipt['error'] = repr(error)
    finally:
        try:
            receipt['after'] = {path: sha(path) for path in pins}
            receipt['effective_after'] = effective()
            if pins:
                assert pins == receipt['after']
        except BaseException as error:
            code = 1
            receipt['Status'] = 'failed'
            receipt['after_check_error'] = repr(error)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        receipt.update({'lock_released_utc': utc(), 'finished_utc': utc(), 'exit': code})
        write(root, 'runtime.json', receipt)
    return code


if '--runtime-admitted' not in sys.argv:
    print(json.dumps({'Status': 'runtime_held', 'exit': 75}))
    raise SystemExit(75)
selection = sys.argv[sys.argv.index('--selection') + 1]
assert selection == 'all'
build_sha = sys.argv[sys.argv.index('--build-sha') + 1]
input_sha = sys.argv[sys.argv.index('--input-sha') + 1]
label = 'first' if selection == 'first' else 'matrix'
if input_sha != INPUTS_SHA:
    label += '-rejection'
root = ROOT / ('artifacts/hac-vcs-aux-observer/' + label + '-001')
if '--inner' in sys.argv:
    raise SystemExit(inner(root, selection, build_sha, input_sha))
root.parent.mkdir(parents=True, exist_ok=True)
root.mkdir()
(root / 'tmp').mkdir()
(root / 'uvcache').mkdir()
unit = 'hac-vcs-aux-' + label + '-20261025-001.service'
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + unit,
        '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=3G',
        '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=175s',
        '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s', '-p', 'LimitCORE=0',
        '--setenv=UV_PYTHON_DOWNLOADS=never', '--setenv=UV_OFFLINE=1', '--setenv=UV_PYTHON=/usr/bin/python3.14', '--setenv=UV_CACHE_DIR=' + str(root / 'uvcache'),
        '/home/jake/.local/bin/uv', 'run', '--no-project', '--offline', '--python', '/usr/bin/python3.14', 'python', str(SCRIPT),
        '--runtime-admitted', '--selection', selection, '--build-sha', build_sha, '--input-sha', input_sha, '--inner']
started = utc()
with (root / 'outer.stdout').open('xb') as stdout, (root / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
write(root, 'outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(),
                                'exit': result.returncode, 'stdout_sha256': sha(root / 'outer.stdout'),
                                'stderr_sha256': sha(root / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(root / 'runtime.json')}))
raise SystemExit(result.returncode)
