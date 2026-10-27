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
OUT = ROOT / 'artifacts/hac-vcs-aux/build-003'
SCRIPT = ROOT / 'verification/hacvcsaux/relink.py'
PLAN = ROOT / 'verification/hacvcsaux/relink-plan.json'
PLAN_SHA = 'f1df0dbeba0dac4075d47089e2faf5156284f8da14352a2c341dc756c68cd1b1'
HOST = ROOT / 'artifacts/hac-vcs-aux/host-003/receipt.json'
CHECKER = ROOT / 'artifacts/hac-vcs-aux/host-003/hacvcsaux'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-relink-20261025-003.service'
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC',
       'TMPDIR': str(OUT / 'tmp'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
       'GOMAXPROCS': '2', 'GOMEMLIMIT': '3GiB', 'UV_PYTHON': '/usr/bin/python3.14',
       'UV_PYTHON_DOWNLOADS': 'never', 'UV_OFFLINE': '1', 'UV_CACHE_DIR': str(OUT / 'uvcache')}


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
    relative = next(line[3:] for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::/'))
    group = pathlib.Path('/sys/fs/cgroup') / relative.lstrip('/')
    result = {'cgroup': str(group), 'pid': os.getpid(), 'parent_pid': os.getppid()}
    for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']:
        result[name] = (group / name).read_text().strip()
    assert [result[name] for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']] == ['200000 100000', '3221225472', '0', '64']
    return result


def execute(label, argv, seconds=20):
    receipt = {'argv': argv, 'environment': ENV, 'started_utc': utc(), 'process_seconds': seconds,
               'stdout_limit': 2 * 1024 * 1024, 'stderr_limit': 65536}
    child = subprocess.Popen(argv, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    receipt['pid'] = child.pid
    deadline = time.monotonic() + seconds
    streams = selectors.DefaultSelector()
    streams.register(child.stdout, selectors.EVENT_READ, 'stdout')
    streams.register(child.stderr, selectors.EVENT_READ, 'stderr')
    counts = {'stdout': 0, 'stderr': 0}
    hashes = {name: hashlib.sha256() for name in counts}
    failure = None
    with (OUT / (label + '.stdout')).open('xb') as output, (OUT / (label + '.stderr')).open('xb') as errors:
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
    write(label + '-receipt.json', receipt)
    return receipt


def dependencies(path, target):
    body = path.read_text().replace('\\\n', '')
    prefix = target + '-dependencies:'
    assert body.startswith(prefix), 'unexpected dependency target'
    members = body[len(prefix):].split()
    assert members and all(not any(character in name for character in ['\\', '$', ':']) for name in members), 'unsupported fixed-profile dependency spelling'
    result = {}
    for name in members:
        lexical = pathlib.Path(name)
        if not lexical.is_absolute():
            lexical = ROOT / lexical
        actual = lexical.resolve(strict=True)
        assert actual.is_file()
        result[str(lexical)] = str(actual)
    return result


def inner():
    receipt = {'Schema': 'hac-vcs-aux-build/v1', 'Status': 'failed', 'started_utc': utc(),
               'effective_before': effective(), 'environment': ENV, 'commands': [],
               'RuntimeHeld': True, 'OracleRunsHeld': True, 'Inputs': {}, 'Objects': {}}
    descriptor = os.open(LOCK, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        receipt.update({'Status': 'deferred_host_lock', 'exit': 75, 'finished_utc': utc()})
        write('build.json', receipt)
        return 75
    receipt['lock_acquired_utc'] = utc()
    code = 1
    bindings = {}
    try:
        assert sha(PLAN) == PLAN_SHA
        plan = json.loads(PLAN.read_text())
        assert plan['Schema'] == 'hac-vcs-aux-relink-plan/v1' and plan['NativeCommandCount'] == 1
        assert plan['RuntimeHeld'] and plan['OracleRunsHeld']
        for name in ['PriorBuild', 'PriorClosure', 'OriginalRecipe', 'CompilerBaseline', 'Checker', 'Loader']:
            selected = plan[name]
            assert sha(selected['Path']) == selected['SHA256'], 'retained pin changed: ' + name
        prior = json.loads(pathlib.Path(plan['PriorBuild']['Path']).read_text())
        frozen = json.loads(pathlib.Path(plan['PriorClosure']['Path']).read_text())
        original = json.loads(pathlib.Path(plan['OriginalRecipe']['Path']).read_text())
        assert prior['Schema'] == 'hac-vcs-aux-build/v1' and prior['Status'] == 'failed' and prior['exit'] == 1
        assert prior['Inputs'] == prior['inputs_after'] == frozen['Inputs']
        assert prior['Objects'] == prior['objects_after'] == plan['RetainedObjects'] and len(prior['Objects']) == 2
        assert len(prior['commands']) == 63 and [i for i, command in enumerate(prior['commands']) if command['exit'] != 0] == [61]
        assert prior['commands'][61]['exit'] == 1 and all(command['failure'] is None for command in prior['commands'])
        assert prior['OriginalObjects'] == plan['OriginalObjects'] == original['OriginalObjects']
        assert prior['SourcePins'] == plan['SourcePins'] == original['SourcePins']
        assert plan['Checker']['Path'] == str(CHECKER)
        assert prior['Inputs'][str(CHECKER)] == plan['Checker']['SHA256']
        assert plan['Loader']['Path'] == '/usr/lib/ld-linux-x86-64.so.2'
        assert plan['Loader']['SHA256'] == 'f5e11cc62f8f2c24dff532982559f4303dba3b17e7450551372a88c8e7ea8757'
        assert prior['Inputs'][plan['Loader']['Path']] == plan['Loader']['SHA256']
        expected_link = []
        old_directory = pathlib.Path(plan['PriorBuild']['Path']).parent
        for value in original['Link']:
            if value in ['{OUT}/oracle.o', '{OUT}/transport.o']:
                value = str(old_directory / pathlib.Path(value).name)
                assert value in prior['Objects']
            if value == '-Wl,--end-group':
                expected_link.append(plan['Loader']['Path'])
            expected_link.append(value)
        assert expected_link == plan['Link'], 'relink differs beyond retained objects, output bindings, and CPU loader'
        receipt['Inputs'].update(prior['Inputs'])
        receipt['Inputs'].update(plan['PreservedLogs'])
        for name in ['PriorBuild', 'PriorClosure', 'OriginalRecipe', 'CompilerBaseline', 'Checker', 'Loader']:
            selected = plan[name]
            receipt['Inputs'][selected['Path']] = selected['SHA256']
        receipt['Inputs'].update({str(PLAN): PLAN_SHA, str(SCRIPT): sha(SCRIPT)})
        receipt['Objects'].update(prior['Objects'])
        receipt['OriginalObjects'] = prior['OriginalObjects']
        receipt['SourcePins'] = prior['SourcePins']
        receipt['CompilerBaselineSHA256'] = prior['CompilerBaselineSHA256']
        assert receipt['CompilerBaselineSHA256'] == plan['CompilerBaseline']['SHA256']
        receipt['Dependencies'] = prior['Dependencies']
        receipt['RelinkProvenance'] = plan
        bindings.update(frozen['Bindings'])

        def check_pins():
            for path, expected in receipt['Inputs'].items():
                assert sha(path) == expected, 'input changed: ' + path
            for path, expected in receipt['Objects'].items():
                assert sha(path) == expected, 'retained object changed: ' + path
            for path, expected in bindings.items():
                assert str(pathlib.Path(path).resolve(strict=True)) == expected, 'header/include binding changed: ' + path

        check_pins()
        for name in ['oracle', 'transport']:
            selected = dependencies(old_directory / (name + '.d'), name)
            assert sorted(set(selected.values())) == receipt['Dependencies'][name]
            for lexical, actual in selected.items():
                assert bindings[lexical] == actual

        def checked(label, argv, seconds=20):
            check_pins()
            before = execute(label + '-compiler-before', [str(CHECKER), '-unit', 'compiler-check'])
            receipt['commands'].append(before)
            assert before['exit'] == 0 and before['failure'] is None
            result = execute(label, argv, seconds)
            receipt['commands'].append(result)
            after = execute(label + '-compiler-after', [str(CHECKER), '-unit', 'compiler-check'])
            receipt['commands'].append(after)
            check_pins()
            assert after['exit'] == 0 and after['failure'] is None
            assert result['exit'] == 0 and result['failure'] is None
            return result

        write('retained-before-link.json', {'Inputs': receipt['Inputs'], 'Objects': receipt['Objects'], 'Bindings': bindings})
        checked('link', [value.replace('{OUT}', str(OUT)) for value in plan['Link']], 30)
        trace = (OUT / 'link.stdout').read_text().splitlines()
        assert trace
        for line in trace:
            member = pathlib.Path(line.strip())
            assert member.is_absolute(), 'unrecognized linker trace record: ' + line
            actual = str(member.resolve(strict=True))
            assert actual in receipt['Inputs'] or actual in receipt['Objects'], 'link used undeclared input: ' + actual
        receipt['LinkTraceSHA256'] = sha(OUT / 'link.stdout')
        receipt['LinkMapSHA256'] = sha(OUT / 'oracle.map')
        oracle = OUT / 'oracle'
        receipt['Output'] = {'Path': str(oracle), 'SHA256': sha(oracle)}
        checked('oracle-elf', [str(CHECKER), '-unit', 'artifact-check', '-oracle', str(oracle), '-oracle-sha', sha(oracle)])
        receipt['OracleELFSHA256'] = sha(OUT / 'oracle-elf.stdout')
        assert sha(oracle) == receipt['Output']['SHA256']
        receipt['Status'] = 'built_outputs_unexecuted'
        code = 0
    except BaseException as error:
        receipt['error'] = repr(error)
    finally:
        try:
            receipt['inputs_after'] = {path: sha(path) for path in receipt['Inputs']}
            receipt['objects_after'] = {path: sha(path) for path in receipt['Objects']}
            receipt['effective_after'] = effective()
            assert receipt['Inputs'] == receipt['inputs_after'] and receipt['Objects'] == receipt['objects_after']
            if 'Output' in receipt:
                receipt['output_after'] = sha(receipt['Output']['Path'])
                assert receipt['output_after'] == receipt['Output']['SHA256']
            assert all(str(pathlib.Path(path).resolve(strict=True)) == expected for path, expected in bindings.items())
        except BaseException as error:
            receipt['after_check_error'] = repr(error)
            code = 1
            receipt['Status'] = 'failed'
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        receipt.update({'lock_released_utc': utc(), 'finished_utc': utc(), 'exit': code})
        write('build.json', receipt)
    return code


if '--link-admitted' not in sys.argv:
    print(json.dumps({'Status': 'relink_held', 'exit': 75}))
    raise SystemExit(75)
if '--inner' in sys.argv:
    raise SystemExit(inner())
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir()
for directory in ['tmp', 'uvcache']:
    (OUT / directory).mkdir()
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT,
        '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=3G',
        '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=175s',
        '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s',
        '--setenv=UV_PYTHON_DOWNLOADS=never', '--setenv=UV_OFFLINE=1', '--setenv=UV_PYTHON=/usr/bin/python3.14', '--setenv=UV_CACHE_DIR=' + str(OUT / 'uvcache'),
        '/home/jake/.local/bin/uv', 'run', '--no-project', '--offline', '--python', '/usr/bin/python3.14', 'python', str(SCRIPT), '--link-admitted', '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(),
                            'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'),
                            'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'build.json')}))
raise SystemExit(result.returncode)
