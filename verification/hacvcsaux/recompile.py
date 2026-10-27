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
OUT = ROOT / 'artifacts/hac-vcs-aux/build-004'
SCRIPT = ROOT / 'verification/hacvcsaux/recompile.py'
PLAN = ROOT / 'verification/hacvcsaux/recompile-plan.json'
PLAN_SHA = '5bc51df352a5938eed3059c956f40a7b5a13afc1bd252d6f38956e54ea062f08'
HOST = ROOT / 'artifacts/hac-vcs-aux/host-003/receipt.json'
CHECKER = ROOT / 'artifacts/hac-vcs-aux/host-003/hacvcsaux'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-recompile-20261025-004.service'
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
        assert plan['Schema'] == 'hac-vcs-aux-recompile-plan/v1' and plan['RuntimeHeld'] and plan['OracleRunsHeld']
        names = ['PriorRelink', 'PriorBuild', 'PriorClosure', 'OriginalRecipe', 'PriorRelinkRecipe',
                 'CompilerBaseline', 'Checker', 'WrongObject', 'TransportObject', 'PublicHeader',
                 'OfficialWrapperHeader', 'StockExports', 'StockExportsReceipt', 'NM']
        for name in names:
            selected = plan[name]
            assert sha(selected['Path']) == selected['SHA256'], 'retained pin changed: ' + name
        prior = json.loads(pathlib.Path(plan['PriorRelink']['Path']).read_text())
        compiled = json.loads(pathlib.Path(plan['PriorBuild']['Path']).read_text())
        frozen = json.loads(pathlib.Path(plan['PriorClosure']['Path']).read_text())
        original = json.loads(pathlib.Path(plan['OriginalRecipe']['Path']).read_text())
        relink = json.loads(pathlib.Path(plan['PriorRelinkRecipe']['Path']).read_text())
        assert prior['Schema'] == 'hac-vcs-aux-build/v1' and prior['Status'] == 'failed' and prior['exit'] == 1
        assert len(prior['commands']) == 3 and [c['exit'] for c in prior['commands']] == [0, 1, 0]
        assert all(c['failure'] is None for c in prior['commands'])
        assert prior['Inputs'] == prior['inputs_after']
        assert prior['Objects'] == prior['objects_after'] == plan['RetainedObjects'] == compiled['Objects']
        assert compiled['Inputs'] == compiled['inputs_after'] == frozen['Inputs']
        assert prior['SourcePins'] == compiled['SourcePins'] == original['SourcePins']
        assert prior['OriginalObjects'] == compiled['OriginalObjects'] == original['OriginalObjects']
        correction = plan['SourceCorrection']
        before, preserved, after = (correction[name] for name in ['Before', 'PreservedBefore', 'After'])
        assert before == prior['SourcePins']['oracle.cpp'] and before['Path'] == after['Path']
        assert before['SHA256'] == preserved['SHA256'] == sha(preserved['Path'])
        assert after['SHA256'] == sha(after['Path'])
        old_bytes = pathlib.Path(preserved['Path']).read_bytes()
        removed = correction['OnlyRemovedBytes'].encode()
        assert removed == b'#include "koi.h"\n' and old_bytes.count(removed) == 1
        assert pathlib.Path(after['Path']).read_bytes() == old_bytes.replace(removed, b'', 1)
        assert prior['Inputs'][after['Path']] == before['SHA256']
        assert plan['Compile'] == original['Units'][0]['Compile']
        assert plan['WrongObject']['SHA256'] == prior['Objects'][plan['WrongObject']['Path']]
        assert plan['TransportObject']['SHA256'] == prior['Objects'][plan['TransportObject']['Path']]
        assert plan['NM']['Path'] == '/usr/bin/nm' and prior['Inputs']['/usr/bin/nm'] == plan['NM']['SHA256']
        assert plan['WrongABI'] == ['/usr/bin/nm', '--undefined-only', '--no-demangle', plan['WrongObject']['Path']]
        assert plan['CorrectedABI'] == ['/usr/bin/nm', '--undefined-only', '--no-demangle', '{OUT}/oracle.o']
        expected_link = ['{OUT}/oracle.o' if value == plan['WrongObject']['Path'] else value for value in relink['Link']]
        assert expected_link == plan['Link']
        assert plan['Checker']['Path'] == str(CHECKER) and prior['Inputs'][str(CHECKER)] == plan['Checker']['SHA256']
        exports = pathlib.Path(plan['StockExports']['Path']).read_text().splitlines()
        for name in plan['OriginalDefinedPublicSymbols']:
            matches = [line for line in exports if line.endswith(' T ' + name)]
            assert len(matches) == 1
            member = 'utils_lstm.cu.o' if name == 'host_lstm_preprocess' else 'thread_pool.cpp.o'
            assert ':' + member + ':' in matches[0]
        official = pathlib.Path(plan['OfficialWrapperHeader']['Path']).read_text()
        assert 'extern "C" {' in official and '#include "koi.h"' in official
        receipt['Inputs'].update(prior['Inputs'])
        receipt['Inputs'][after['Path']] = after['SHA256']
        receipt['Inputs'][preserved['Path']] = preserved['SHA256']
        receipt['Inputs'].update(plan['PreservedRelinkLogs'])
        for name in names:
            selected = plan[name]
            receipt['Inputs'][selected['Path']] = selected['SHA256']
        receipt['Inputs'].update({str(PLAN): PLAN_SHA, str(SCRIPT): sha(SCRIPT)})
        receipt['Objects'][plan['TransportObject']['Path']] = plan['TransportObject']['SHA256']
        receipt['OriginalObjects'] = prior['OriginalObjects']
        receipt['SourcePins'] = dict(prior['SourcePins'])
        receipt['SourcePins']['oracle.cpp'] = after
        receipt['CompilerBaselineSHA256'] = prior['CompilerBaselineSHA256']
        assert receipt['CompilerBaselineSHA256'] == plan['CompilerBaseline']['SHA256']
        receipt['Dependencies'] = compiled['Dependencies']
        receipt['SourceCorrectionProvenance'] = plan
        bindings.update(frozen['Bindings'])

        def check_pins():
            for path, expected in receipt['Inputs'].items():
                assert sha(path) == expected, 'input changed: ' + path
            for path, expected in receipt['Objects'].items():
                assert sha(path) == expected, 'retained object changed: ' + path
            for path, expected in bindings.items():
                assert str(pathlib.Path(path).resolve(strict=True)) == expected, 'header/include binding changed: ' + path

        check_pins()
        old_directory = pathlib.Path(plan['PriorBuild']['Path']).parent
        for name in ['oracle', 'transport']:
            selected = dependencies(old_directory / (name + '.d'), name)
            assert sorted(set(selected.values())) == receipt['Dependencies'][name]
            for lexical, actual in selected.items():
                assert bindings[lexical] == actual

        def checked(label, argv, seconds=20):
            check_pins()
            before_check = execute(label + '-compiler-before', [str(CHECKER), '-unit', 'compiler-check'])
            receipt['commands'].append(before_check)
            assert before_check['exit'] == 0 and before_check['failure'] is None
            result = execute(label, argv, seconds)
            receipt['commands'].append(result)
            after_check = execute(label + '-compiler-after', [str(CHECKER), '-unit', 'compiler-check'])
            receipt['commands'].append(after_check)
            check_pins()
            assert after_check['exit'] == 0 and after_check['failure'] is None
            assert result['exit'] == 0 and result['failure'] is None
            return result

        def public_abi(label, expected):
            symbols = []
            for line in (OUT / (label + '.stdout')).read_text().splitlines():
                fields = line.split()
                assert len(fields) == 2 and fields[0] in ['U', 'w', 'v'], 'unsupported nm undefined record'
                symbols.append(fields[1])
            required = set(plan['RequiredUndefinedPublicSymbols'])
            assert required == {'host_lstm_preprocess', 'koi_create_thread_pool', 'koi_destroy_thread_pool'}
            mangled = [name for name in symbols if 'host_lstm_preprocess' in name and name != 'host_lstm_preprocess']
            accepted = required.issubset(symbols) and not mangled
            if expected:
                assert accepted, 'public host/thread-pool C ABI regression'
            else:
                assert not accepted and 'host_lstm_preprocess' not in symbols and len(mangled) == 1
                assert mangled[0].startswith('_Z20host_lstm_preprocess')
                assert required.difference({'host_lstm_preprocess'}).issubset(symbols)
            return {'Accepted': accepted, 'RequiredSymbols': sorted(required), 'MangledHostSymbols': mangled}

        write('retained-before-compile.json', {'Inputs': receipt['Inputs'], 'Objects': receipt['Objects'], 'Bindings': bindings})
        checked('wrong-abi', plan['WrongABI'])
        receipt['PublicSymbolRegressionNegative'] = public_abi('wrong-abi', False)
        checked('oracle-compile', [value.replace('{OUT}', str(OUT)) for value in plan['Compile']], 120)
        selected = dependencies(OUT / 'oracle.d', 'oracle')
        assert sorted(set(selected.values())) == receipt['Dependencies']['oracle'], 'corrected source selected different headers'
        assert all(bindings[lexical] == actual for lexical, actual in selected.items())
        receipt['Objects'][str(OUT / 'oracle.o')] = sha(OUT / 'oracle.o')
        checked('corrected-abi', [value.replace('{OUT}', str(OUT)) for value in plan['CorrectedABI']])
        receipt['PublicSymbolRegressionPositive'] = public_abi('corrected-abi', True)
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


if '--compile-admitted' not in sys.argv:
    print(json.dumps({'Status': 'recompile_held', 'exit': 75}))
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
        '/home/jake/.local/bin/uv', 'run', '--no-project', '--offline', '--python', '/usr/bin/python3.14', 'python', str(SCRIPT), '--compile-admitted', '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(),
                            'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'),
                            'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'build.json')}))
raise SystemExit(result.returncode)
