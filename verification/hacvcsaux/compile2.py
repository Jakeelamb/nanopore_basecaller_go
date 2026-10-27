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
OUT = ROOT / 'artifacts/hac-vcs-aux/build-002'
SCRIPT = ROOT / 'verification/hacvcsaux/compile2.py'
PLAN = ROOT / 'verification/hacvcsaux/oracle-plan.json'
PLAN_SHA = 'bf60a098af27bd82c9113c62a60569521d1d562c5b626b9166661ff8eae9b0c7'
HOST = ROOT / 'artifacts/hac-vcs-aux/host-003/receipt.json'
CHECKER = ROOT / 'artifacts/hac-vcs-aux/host-003/hacvcsaux'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-compile-20261025-002.service'
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


def inner(host_sha):
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
        assert sha(PLAN) == PLAN_SHA and sha(HOST) == host_sha
        plan = json.loads(PLAN.read_text())
        host = json.loads(HOST.read_text())
        assert plan['Schema'] == 'hac-vcs-aux-oracle-plan/v1' and plan['RuntimeHeld'] and plan['OracleRunsHeld']
        assert host['exit'] == 0 and host['checker_sha256'] == sha(CHECKER)
        assert host['source_after'] == host['source_after_format']
        receipt['Inputs'].update(plan['InitialInputs'])
        receipt['Inputs'].update(host['selected_inputs'])
        receipt['Inputs'].update(host['tools_before'])
        receipt['Inputs'].update({str(PLAN): PLAN_SHA, str(HOST): host_sha, str(CHECKER): host['checker_sha256'], str(SCRIPT): sha(SCRIPT)})
        receipt['OriginalObjects'] = plan['OriginalObjects']
        receipt['SourcePins'] = plan['SourcePins']
        receipt['CompilerBaselineSHA256'] = plan['CompilerBaseline']['SHA256']
        receipt['Dependencies'] = {}
        bindings.update(plan['IncludeBindings'])

        def check_pins():
            for path, expected in receipt['Inputs'].items():
                assert sha(path) == expected, 'input pin changed: ' + path
            for path, expected in receipt['Objects'].items():
                assert sha(path) == expected, 'retained object changed: ' + path
            for path, expected in bindings.items():
                assert str(pathlib.Path(path).resolve(strict=True)) == expected, 'path binding changed: ' + path

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

        for index, query in enumerate(plan['Queries']):
            label = 'query-%02d' % index
            checked(label, query['Argv'])
            value = (OUT / (label + '.stdout')).read_text().strip()
            assert value and '\n' not in value and '\r' not in value
            selected = pathlib.Path(value)
            if not selected.is_absolute():
                assert query['Kind'] == 'program' and '/' not in value
                selected = pathlib.Path('/usr/bin') / value
            assert str(selected.resolve(strict=True)) == query['Expected']['Path']
            assert sha(selected) == query['Expected']['SHA256']

        header_roots = [pathlib.Path('/usr/include'), pathlib.Path('/opt/cuda').resolve(),
                        pathlib.Path(plan['Driver']['Path']).parents[1],
                        pathlib.Path(next(path for path in plan['IncludeBindings'] if 'libkoi-' in path)).resolve(),
                        ROOT / 'verification/hacvcsaux/native']
        for unit in plan['Units']:
            label = unit['Name'] + '-dependencies'
            checked(label, unit['Dependencies'])
            selected = dependencies(OUT / (label + '.stdout'), unit['Name'])
            for lexical, actual in selected.items():
                path = pathlib.Path(actual)
                assert any(path == base or base in path.parents for base in header_roots), 'dependency outside fixed source/header roots: ' + actual
                bindings[lexical] = actual
                receipt['Inputs'][actual] = sha(path)
            receipt['Dependencies'][unit['Name']] = sorted(set(selected.values()))
        check_pins()
        write('before-compile.json', {'Inputs': receipt['Inputs'], 'Dependencies': receipt['Dependencies'], 'Bindings': bindings})
        for unit in plan['Units']:
            argv = [value.replace('{OUT}', str(OUT)) for value in unit['Compile']]
            checked(unit['Name'] + '-compile', argv, 120)
            selected = dependencies(OUT / (unit['Name'] + '.d'), unit['Name'])
            assert sorted(set(selected.values())) == receipt['Dependencies'][unit['Name']], 'M/MD dependency closure differs'
            for lexical, actual in selected.items():
                assert bindings.get(lexical) == actual, 'M/MD lexical binding differs'
            object_path = OUT / (unit['Name'] + '.o')
            receipt['Objects'][str(object_path)] = sha(object_path)
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
        assert sha(oracle) == receipt['Output']['SHA256'], 'oracle changed during static ELF readback'
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


if '--compiler-admitted' not in sys.argv:
    print(json.dumps({'Status': 'compiler_held', 'exit': 75}))
    raise SystemExit(75)
host_sha = sys.argv[sys.argv.index('--host-sha') + 1]
if '--inner' in sys.argv:
    raise SystemExit(inner(host_sha))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir()
for directory in ['tmp', 'uvcache']:
    (OUT / directory).mkdir()
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT,
        '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=3G',
        '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=175s',
        '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s',
        '--setenv=UV_PYTHON_DOWNLOADS=never', '--setenv=UV_OFFLINE=1', '--setenv=UV_PYTHON=/usr/bin/python3.14', '--setenv=UV_CACHE_DIR=' + str(OUT / 'uvcache'), '/home/jake/.local/bin/uv',
        'run', '--no-project', '--offline', '--python', '/usr/bin/python3.14', 'python', str(SCRIPT), '--compiler-admitted', '--host-sha', host_sha, '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(),
                            'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'),
                            'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'build.json')}))
raise SystemExit(result.returncode)
