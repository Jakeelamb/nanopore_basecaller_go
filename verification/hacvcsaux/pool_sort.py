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
OUT = ROOT / 'artifacts/hac-vcs-aux/pool-sort-001'
OBJECT = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go/artifacts/parity-resume-review-20261024/hac-vcs-native-boundary/body-001/utils_lstm.cpp.o')
SCRIPT = ROOT / 'verification/hacvcsaux/pool_sort.py'
POOL = ROOT / 'artifacts/hac-vcs-aux/frontier-001/thread_pool.cpp.o'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-pool-sort-20261025-001.service'
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC'}
TOOLS = [pathlib.Path('/usr/bin/objdump')]
SYMBOLS = [
    ('merge-sort', '_ZSt24__merge_sort_with_bufferIN9__gnu_cxx17__normal_iteratorIPN3koi4lstm12_GLOBAL__N_110ChunkChainESt6vectorIS5_SaIS5_EEEES6_NS0_5__ops15_Iter_comp_iterISt7greaterIS5_EEEEvT_SG_T0_T1_.isra.0', OBJECT),
    ('pool-create', 'koi_create_thread_pool', POOL),
    ('pool-destroy', 'koi_destroy_thread_pool', POOL),
]

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
    assert [result[name] for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']] == ['200000 100000', '268435456', '0', '64']
    return result

def run(label, argv, stdout, cap):
    receipt = {'argv': argv, 'environment': ENV, 'started_utc': utc(), 'stdout_limit': cap, 'stderr_limit': 65536, 'process_seconds': 20}
    child = subprocess.Popen(argv, cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    receipt['pid'] = child.pid
    deadline = time.monotonic() + 20
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
    assert receipt['exit'] == 0 and failure is None
    return receipt

def inner():
    receipt = {'started_utc': utc(), 'effective_before': effective(), 'environment': ENV, 'commands': []}
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
        receipt['before'] = {str(path): sha(path) for path in [OBJECT, POOL, SCRIPT] + TOOLS}
        assert receipt['before'][str(OBJECT)] == '7598a9c75f9b4d3f3936e3871a31155a8e4b63c417017b7f5b2132db96051243'
        assert receipt['before'][str(POOL)] == '89cdf4ea0968f38e557787203b8a8b180dcac74c6df077bf89bf18d837b13f2d'
        assert receipt['before']['/usr/bin/objdump'] == '9273687d5b7323cb103300f4facb847276657995ea6d5e5e6e25c97f4fc4594e'
        for label, symbol, member in SYMBOLS:
            receipt['commands'].append(run(label, ['/usr/bin/objdump', '-dr', '--no-show-raw-insn', '--disassemble=' + symbol, str(member)], OUT / (label + '.stdout'), 1024 * 1024))
        receipt['after'] = {str(path): sha(path) for path in [OBJECT, POOL, SCRIPT] + TOOLS}
        assert receipt['before'] == receipt['after']
        receipt['effective_after'] = effective()
        code = 0
    except BaseException as error:
        receipt['error'] = repr(error)
    finally:
        try:
            receipt['after'] = {str(path): sha(path) for path in [OBJECT, POOL, SCRIPT] + TOOLS}
            receipt['effective_after'] = effective()
            assert receipt['before'] == receipt['after']
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
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT, '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=256M', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=75s', '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s', '/home/jake/.local/bin/uv', 'run', '--no-project', 'python', str(SCRIPT), '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=80)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(), 'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'), 'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'receipt.json')}))
raise SystemExit(result.returncode)
