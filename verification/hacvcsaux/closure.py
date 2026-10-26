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
OUT = ROOT / 'artifacts/hac-vcs-aux/closure-001'
MEMBERS = pathlib.Path('/home/jake/Projects/nanopore_basecaller_go/artifacts/parity-resume-review-20261024/hac-vcs-native-boundary/body-001')
OBJECTS = [MEMBERS / 'utils_lstm.cpp.o', MEMBERS / 'utils_lstm.cu.o']
ARCHIVE = pathlib.Path('/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/libkoi-0.7.5-Linux-x86_64-cuda-13.0/libkoi-0.7.5-Linux-x86_64-cuda-13.0/lib/libkoi.a')
SCRIPT = ROOT / 'verification/hacvcsaux/closure.py'
LOCK = '/home/jake/Projects/nanopore_basecaller/results/stock/host.lock'
UNIT = 'hac-vcs-aux-closure-20261025-001.service'
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC', 'TMPDIR': str(OUT / 'tmp')}
TOOLS = [pathlib.Path('/usr/bin/nm'), pathlib.Path('/usr/bin/readelf'), pathlib.Path('/usr/bin/objdump')]
COMMANDS = [
    ('cpp-undefined', ['/usr/bin/nm', '--undefined-only', '--no-demangle', str(OBJECTS[0])]),
    ('cu-undefined', ['/usr/bin/nm', '--undefined-only', '--no-demangle', str(OBJECTS[1])]),
    ('pool-members', ['/usr/bin/nm', '-A', '--no-demangle', '--defined-only', '--extern-only', str(ARCHIVE)]),
    ('cpp-metadata', ['/usr/bin/readelf', '-SW', '-sW', '-rW', str(OBJECTS[0])]),
    ('cu-metadata', ['/usr/bin/readelf', '-SW', '-sW', '-rW', str(OBJECTS[1])]),
    ('cpp-cst16', ['/usr/bin/objdump', '-s', '-j', '.rodata.cst16', str(OBJECTS[0])]),
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
        receipt['before'] = {str(path): sha(path) for path in OBJECTS + [ARCHIVE, SCRIPT] + TOOLS}
        assert receipt['before'][str(OBJECTS[0])] == '7598a9c75f9b4d3f3936e3871a31155a8e4b63c417017b7f5b2132db96051243'
        assert receipt['before'][str(OBJECTS[1])] == 'ce998c4411202c7c1c1d6240bad5677dfb8c1d69118355bf9affef6170045117'
        assert receipt['before'][str(ARCHIVE)] == '17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f'
        assert receipt['before']['/usr/bin/nm'] == 'b16707aadd1c1953f51fa32d462b2d463a54031f7a743d63e8333c6bf3985abd'
        assert receipt['before']['/usr/bin/readelf'] == 'a72f12f3dd8c560554a3ba818bcecbda9178db5befbcc64d91bbb6a241fc21bc'
        assert receipt['before']['/usr/bin/objdump'] == '9273687d5b7323cb103300f4facb847276657995ea6d5e5e6e25c97f4fc4594e'
        for label, argv in COMMANDS:
            receipt['commands'].append(run(label, argv, OUT / (label + '.stdout'), 2 * 1024 * 1024))
        receipt['after'] = {str(path): sha(path) for path in OBJECTS + [ARCHIVE, SCRIPT] + TOOLS}
        assert receipt['before'] == receipt['after']
        receipt['effective_after'] = effective()
        code = 0
    except BaseException as error:
        receipt['error'] = repr(error)
    finally:
        try:
            receipt['after'] = {str(path): sha(path) for path in OBJECTS + [ARCHIVE, SCRIPT] + TOOLS}
            receipt['effective_after'] = effective()
            if receipt.get('before') != receipt['after']:
                code = 1
                receipt['pin_failure'] = 'before_after_pins_differ'
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
(OUT / 'tmp').mkdir()
argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT, '--working-directory=' + str(ROOT), '-p', 'CPUQuota=200%', '-p', 'MemoryMax=256M', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=64', '-p', 'RuntimeMaxSec=135s', '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=2s', '/home/jake/.local/bin/uv', 'run', '--no-project', 'python', str(SCRIPT), '--inner']
started = utc()
with (OUT / 'outer.stdout').open('xb') as stdout, (OUT / 'outer.stderr').open('xb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=140)
write('outer-receipt.json', {'argv': argv, 'started_utc': started, 'finished_utc': utc(), 'exit': result.returncode, 'stdout_sha256': sha(OUT / 'outer.stdout'), 'stderr_sha256': sha(OUT / 'outer.stderr'), 'script_sha256': sha(SCRIPT)})
print(json.dumps({'exit': result.returncode, 'receipt': str(OUT / 'receipt.json')}))
raise SystemExit(result.returncode)
