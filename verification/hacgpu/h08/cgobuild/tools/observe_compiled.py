#!/usr/bin/env python3
"""Independent compiled-observation launcher; run under the documented 600s watchdog."""
import argparse
import datetime
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ENV = ['CGO_ENABLED=0', 'GOTOOLCHAIN=local', 'GOWORK=off', 'GOPROXY=off',
       'GOSUMDB=off', 'GOENV=off', 'GOMAXPROCS=2', 'GOMEMLIMIT=3GiB']
LOCK = Path('/home/jake/Projects/nanopore_basecaller/results/stock/host.lock')
ALGORITHMS = [
    {'Path': '/usr/lib/go/src/cmd/link/internal/ld/fips140.go',
     'SHA256': 'f214648c1a2ec04ea5483812ceec13cec96aa1a1acd9f95d2882b6373ef432c5'},
    {'Path': '/usr/lib/go/src/cmd/internal/obj/fips140.go',
     'SHA256': '25fba80147d77aa812720b5c4ec71304b1547491ff59f0221c12a3c504236137'}]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def decode(body):
    return json.loads(body, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def keys(value, names):
    require(isinstance(value, dict) and set(value) == set(names.split()), 'wire fields differ')


def canonical(path, existing=True):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=existing) == p, 'canonical absolute path required')
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = canonical(path)
    require(p.is_file(), 'regular file required')
    return {'Path': str(p), 'SHA256': digest(p)}


def check_pin(p):
    keys(p, 'Path SHA256')
    require(isinstance(p['SHA256'], str) and re.fullmatch('[0-9a-f]{64}', p['SHA256']), 'invalid digest')
    require(pin(p['Path']) == p, 'independent digest mismatch')


def read_pin(p):
    check_pin(p)
    body = Path(p['Path']).read_bytes()
    require(hashlib.sha256(body).hexdigest() == p['SHA256'], 'metadata changed during read')
    return decode(body)


def write(path, value):
    # Exclusive creation and read-only final bytes; never repair an old attempt.
    path = Path(path)
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    temporary.chmod(0o400)
    os.link(temporary, path)  # Atomic publication, and EEXIST refuses reuse.
    temporary.unlink()
    return pin(path)


def fresh(path, root):
    p = canonical(path, False)
    base = root / 'artifacts/tensor'
    require(p != base and p.is_relative_to(base), 'directory must be beneath ObserverRoot/artifacts/tensor')
    require(not p.exists() and not p.is_symlink(), 'output reuse refused')
    return p


def select(request):
    q = read_pin(request)
    keys(q, 'Schema Role Retained ObserverController ObserverRoot AlgorithmReferences OutputDirectory')
    require(q['Schema'] == 'nanopore-h08-observation-request/v1' and q['Role'] == 'h08-command', 'request schema/role differs')
    keys(q['Retained'], 'Producer Handoff Outer Cleanup FailureStderr Profile')
    require(q['Retained']['Profile'] == 'saved005-final-reader-v1', 'retained profile differs')
    keys(q['Retained']['Producer'], 'Receipt Plan Controller Queries Output')
    # The Go observer owns full retained proof and Controller validation.
    require(q['AlgorithmReferences'] == ALGORITHMS, 'algorithm selectors differ')
    root = canonical(q['ObserverRoot'])
    require(root.is_dir(), 'ObserverRoot directory required')
    fresh(q['OutputDirectory'], root)
    c = read_pin(q['ObserverController'])
    keys(c, 'Schema Executable Inputs Snapshots Tools Lists Build')
    require(c['Schema'] == 'nanopore-h08-cgo-interposer/v1', 'controller schema differs')
    for name in ['observation.go', 'cmd/main.go']:
        require(str(root / 'verification/hacgpu/h08/cgobuild' / name) in c['Inputs'], 'selected observer sources absent')
    check_pin(c['Executable'])
    return q, c


def observer_args(executable, request, output):
    return [executable['Path'], '-mode', 'observe-compiled', '-observation-request',
            request['Path'], '-observation-request-sha', request['SHA256'],
            '-out', output, '-host-admitted']


def effective():
    raw = Path('/proc/self/cgroup').read_text()
    require(re.fullmatch(r'0::/[^\r\n]*\n?', raw), 'single cgroup-v2 line required')
    base = Path('/sys/fs/cgroup')
    group = base / raw.strip()[3:].lstrip('/')
    require('..' not in group.parts, 'invalid cgroup')
    values = {name: [] for name in ['cpu.max', 'memory.max', 'memory.swap.max', 'pids.max']}
    rows = []
    for p in [group, *group.parents]:
        if not p.is_relative_to(base):
            break
        row = {'Path': str(p)}
        for name in values:
            file = p / name
            if p == base and not file.exists():
                continue
            text = file.read_text().strip()
            row[name] = text
            if name == 'cpu.max':
                quota, period = text.split()
                if quota != 'max':
                    values[name].append(Fraction(int(quota), int(period)))
            elif text != 'max':
                values[name].append(int(text))
        rows.append(row)
    maxima = [2, 3 << 30, 0, 64]
    require(all(values.values()) and all(0 <= min(v) <= limit for v, limit in zip(values.values(), maxima)),
            'effective <=2CPU/3GiB/swap0/Tasks64 required')
    return rows


def identity(pid):
    base = Path('/proc') / str(pid)
    fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
    require(int(fields[2]) == pid, 'observer is not group leader')
    raw = (base / 'cgroup').read_text()
    require(re.fullmatch(r'0::/[^\r\n]*\n?', raw), 'observer cgroup-v2 identity required')
    return {'PID': pid, 'Start': fields[19], 'Cgroup': raw, 'Group': True}


def verify_live(process, executable, argv, root, unit):
    pid = process['PID']
    require(identity(pid) == process, 'observer identity changed')
    base = Path('/proc') / str(pid)
    require(os.readlink(base / 'exe') == executable['Path'] and digest(base / 'exe') == executable['SHA256'], 'foreign executable')
    require((base / 'cmdline').read_bytes().split(b'\0')[:-1] == [s.encode() for s in argv], 'foreign argv')
    require((base / 'environ').read_bytes().split(b'\0')[:-1] == [s.encode() for s in ENV], 'foreign environment')
    require(os.readlink(base / 'cwd') == str(root), 'foreign cwd')
    require(process['Cgroup'].rstrip('\n').endswith('/' + unit), 'foreign cgroup')
    require(identity(pid) == process, 'observer changed while inspected')


def verify_inner(process, python, argv, unit):
    base = Path('/proc') / str(process['PID'])
    require(identity(process['PID']) == process and
            process['Cgroup'].rstrip('\n').endswith('/' + unit), 'foreign inner service')
    require(os.readlink(base / 'exe') == python['Path'] and
            digest(base / 'exe') == python['SHA256'], 'foreign inner interpreter')
    require((base / 'cmdline').read_bytes().split(b'\0')[:-1] ==
            [v.encode() for v in argv], 'foreign inner argv')


def absent(process):
    try:
        same = identity(process['PID']) == process
    except FileNotFoundError:
        same = False
    require(not same, 'original PID/start identity remains')
    # Read only: never signal a numeric group which might have been reused.
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        require(int(fields[2]) != process['PID'], 'owned process group not empty (or identity ambiguous)')
    group = Path('/sys/fs/cgroup') / process['Cgroup'].strip()[3:].lstrip('/')
    require(not group.exists(), 'owned cgroup remains')


def command(argv, directory, label, timeout=8):
    out, err = directory / (label + '.stdout'), directory / (label + '.stderr')
    r = dict(command=argv, environment_overrides=[], started_utc=utc(), exit_code=-1,
             stdout=str(out), stderr=str(err))
    failure = None
    with out.open('xb') as a, err.open('xb') as b:
        try:
            r['exit_code'] = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=a, stderr=b,
                                             timeout=timeout, check=False).returncode
        except Exception as exc:
            failure = exc
    r.update(finished_utc=utc(), stdout_sha256=digest(out), stderr_sha256=digest(err))
    write(directory / (label + '.command.json'), r)
    if failure:
        raise failure
    require(r['exit_code'] == 0, label + ' command failed')
    return r


def properties(raw):
    body = Path(raw['stdout']).read_text()
    require(body.endswith('\n'), 'cleanup property newline absent')
    result = pairs([line.split('=', 1) for line in body[:-1].split('\n')])
    require(result == dict(ActiveState='inactive', SubState='dead', MainPID='0', ControlGroup=''), 'cleanup properties differ')


def finish(directory, request, q, executable, process, receipt, raw, unit):
    fields = 'command environment_overrides started_utc finished_utc exit_code stdout stderr stdout_sha256 stderr_sha256'
    keys(receipt, fields)
    keys(raw, fields)
    require(receipt['exit_code'] == 0, 'observer failed')
    require(raw['exit_code'] == 0 and raw['environment_overrides'] == [] and
            raw['command'] == ['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'ControlGroup'], 'cleanup command differs')
    start = datetime.datetime.fromisoformat(receipt['started_utc'])
    end = datetime.datetime.fromisoformat(receipt['finished_utc'])
    clean_start = datetime.datetime.fromisoformat(raw['started_utc'])
    clean_end = datetime.datetime.fromisoformat(raw['finished_utc'])
    require(start.utcoffset() == datetime.timedelta(0) and 0 <= (end - start).total_seconds() <= 600 and
            end <= clean_start <= clean_end, 'execution/cleanup clocks differ')
    for name in ['stdout', 'stderr']:
        require(digest(raw[name]) == raw[name + '_sha256'], 'cleanup raw log changed')
    require(receipt['command'] == observer_args(executable, request, q['OutputDirectory']) and
            receipt['environment_overrides'] == ENV, 'observer command differs')
    for name in ['stdout', 'stderr']:
        require(digest(receipt[name]) == receipt[name + '_sha256'], 'raw log changed')
    observation = decode(Path(receipt['stdout']).read_bytes())
    keys(observation, 'Path SHA256')
    require(observation['Path'] == str(Path(q['OutputDirectory']) / 'observation.json'), 'foreign observation output')
    check_pin(observation)
    properties(raw)
    cleanup = write(directory / 'cleanup.json', dict(
        Schema='nanopore-h08-observer-cleanup/v1', UTC=utc(), Unit=unit, Process=process,
        PIDAbsent=True, GroupAbsent=True, CgroupAbsent=True, RawCommand=raw))
    execution = write(directory / 'execution.json', dict(
        Schema='nanopore-h08-observer-execution/v1', Request=request, Observation=observation,
        ObserverController=q['ObserverController'], Executable=executable, Process=process,
        Cwd=q['ObserverRoot'], Command=receipt, Cleanup=cleanup))
    selection = write(directory / 'selection.json', dict(
        Schema='nanopore-h08-observed-selection/v1', Request=request, Execution=execution))
    return dict(Observation=observation, Selection=selection, Execution=execution, Cleanup=cleanup)


def inner(config_pin):
    config = read_pin(config_pin)
    directory = Path(config['Directory'])
    child, receipt, failure = None, None, None
    try:
        effective_limits = effective()
        for runtime_pin in config['Runtime'].values():
            check_pin(runtime_pin)
        write(directory / 'inner-ready.json', identity(os.getpid()))
        arm_end = time.monotonic() + 5
        while not (directory / 'arm').exists() and time.monotonic() < arm_end:
            time.sleep(0.02)
        require((directory / 'arm').exists(), 'parent did not arm private stage')
        q, controller = select(config['Request'])
        executable = controller['Executable']
        require(executable == config['Executable'], 'inner executable selection differs')
        argv = observer_args(executable, config['Request'], q['OutputDirectory'])
        write(directory / 'inner-effective.json', effective_limits)
        write(directory / 'observer.intent.json', dict(Args=argv, Environment=ENV, Cwd=q['ObserverRoot'], Executable=pin(executable['Path'])))
        out, err = directory / 'observer.stdout', directory / 'observer.stderr'
        receipt = dict(command=argv, environment_overrides=ENV, started_utc=utc(), exit_code=-1,
                       stdout=str(out), stderr=str(err))
        with out.open('xb') as a, err.open('xb') as b:
            require(time.monotonic() < config['StartupDeadline'], 'observer startup budget exhausted')
            child = subprocess.Popen(argv, cwd=q['ObserverRoot'], env=dict(s.split('=', 1) for s in ENV),
                                     stdin=subprocess.DEVNULL, stdout=a, stderr=b, start_new_session=True)
            write(directory / 'spawn.json', identity(child.pid))
            receipt['exit_code'] = child.wait(timeout=550)
    except BaseException as exc:
        failure = str(exc)
    finally:
        # Only an unreaped Popen child can be signalled directly. Descendant
        # cleanup belongs to systemd's bounded control-group stop.
        if child is not None and child.poll() is None:
            child.kill()
            try:
                receipt['exit_code'] = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                failure = (failure or '') + '; child cleanup wait expired'
        if child is not None and child.returncode is not None:
            receipt['exit_code'] = child.returncode
        if receipt:
            receipt.update(finished_utc=utc(), stdout_sha256=digest(receipt['stdout']), stderr_sha256=digest(receipt['stderr']))
            write(directory / 'observer.command.json', receipt)
        write(directory / 'inner-result.json', dict(Error=failure))
    # Keep systemd-run's --wait unit reference alive until the parent stops that
    # owned waiter. Then the service may exit and the parent can inspect dead state.
    end = time.monotonic() + 10
    while not (directory / 'release').exists() and time.monotonic() < end:
        time.sleep(0.02)
    return 1 if failure else 0


def launch(request, receipt_path):
    started = time.monotonic()
    directory, waiter, stopped, lock, service_owned = None, None, False, None, False
    failure, summary, process, inner_process = None, None, None, None
    service_requested, cleanup_confirmed = False, False
    unit = 'h08-observe-' + uuid.uuid4().hex + '.service'
    try:
        outer_limits = effective()  # External envelope before bulk pins or children.
        q, controller = select(request)
        root = Path(q['ObserverRoot'])
        directory = fresh(receipt_path, root)
        output = Path(q['OutputDirectory'])
        require(not output.is_relative_to(directory) and not directory.is_relative_to(output), 'receipt/output overlap')
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700)
        executable = controller['Executable']
        config = dict(Request=request, Executable=executable, Directory=str(directory),
                      StartupDeadline=started + 10)
        runtime = {name: pin(path) for name, path in dict(
            Script=str(Path(__file__).resolve()), Python=str(Path(sys.executable).resolve()),
            SystemdRun='/usr/bin/systemd-run', Systemctl='/usr/bin/systemctl').items()}
        config['Runtime'] = runtime
        config_pin = write(directory / 'config.json', config)
        write(directory / 'outer-intent.json', dict(Request=request, SelectedRequest=q,
              Executable=executable, Runtime=runtime, Effective=outer_limits,
              Unit=unit, ExternalSeconds=600, ServiceSeconds=575, ChildSeconds=550,
              ObserverArgs=observer_args(executable, request, q['OutputDirectory']), Environment=ENV,
              OriginalFailedReceipt=q['Retained']['Producer']['Receipt']))
        require(LOCK.resolve(strict=True) == LOCK, 'canonical stock lock required')
        lock = os.open(LOCK, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stat = os.fstat(lock)
        write(directory / 'lock-held.json', dict(Path=str(LOCK), Open='O_RDONLY',
              Flock='LOCK_EX|LOCK_NB', Device=stat.st_dev, Inode=stat.st_ino, UTC=utc()))
        require(time.monotonic() - started < 10, 'service startup budget exhausted')
        argv = ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--unit=' + unit,
                '--property=RuntimeMaxSec=575s', '--property=TimeoutStartSec=5s',
                '--property=TimeoutStopSec=5s', '--property=KillMode=control-group',
                '--property=SendSIGKILL=yes', '--property=CPUQuota=200%',
                '--property=MemoryMax=3G', '--property=MemorySwapMax=0', '--property=TasksMax=64',
                runtime['Python']['Path'], '-I', runtime['Script']['Path'],
                '--_inner', config_pin['Path'], '--_inner-sha256', config_pin['SHA256']]
        write(directory / 'service-intent.json', dict(Args=argv, UTC=utc()))
        with (directory / 'service.stdout').open('xb') as a, (directory / 'service.stderr').open('xb') as b:
            require(time.monotonic() < started + 10, 'service startup budget exhausted')
            service_requested = True
            waiter = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=a, stderr=b, start_new_session=True)
            # Ownership is acknowledged only after this exact transient unit starts
            # our pinned inner process and writes its config-bound spawn record.
            end = started + 570
            process, witnessed = None, False
            while time.monotonic() < end:
                if not service_owned:
                    require(time.monotonic() < started + 10, 'inner handshake startup budget exhausted')
                ready = directory / 'inner-ready.json'
                if ready.exists() and not service_owned:
                    inner_process = decode(ready.read_bytes())
                    verify_inner(inner_process, runtime['Python'], argv[14:], unit)
                    service_owned = True
                    write(directory / 'independent-inner.json', inner_process)
                    require(time.monotonic() < started + 10, 'parent arm startup budget exhausted')
                    write(directory / 'arm', {})
                spawn = directory / 'spawn.json'
                if spawn.exists() and not witnessed:
                    process = decode(spawn.read_bytes())
                    verify_live(process, executable, observer_args(executable, request, q['OutputDirectory']), root, unit)
                    write(directory / 'independent-process.json', process)
                    service_owned, witnessed = True, True
                if (directory / 'inner-result.json').exists():
                    break
                require(waiter.poll() is None, 'service launcher exited before completion')
                time.sleep(0.02)
            require((directory / 'inner-result.json').exists(), 'parent wait deadline exceeded')
            require(witnessed, 'actual observer was not independently witnessed')
            # Popen remains unreaped and alive, so this PID cannot have been reused.
            require(waiter.poll() is None, 'unit reference waiter already exited')
            os.kill(waiter.pid, signal.SIGSTOP)
            stopped = True
            write(directory / 'release', {})
            end = min(started + 585, time.monotonic() + 8)
            while time.monotonic() < end:
                group = Path('/sys/fs/cgroup') / process['Cgroup'].strip()[3:].lstrip('/')
                if not group.exists():
                    break
                time.sleep(0.02)
            absent(process)
            raw = command(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'ControlGroup'], directory, 'cleanup')
            properties(raw)
            require(decode((directory / 'inner-result.json').read_bytes()) == {'Error': None}, 'inner stage failed')
            receipt = decode((directory / 'observer.command.json').read_bytes())
            # Reauthenticate independent selectors after exit; no producer replay.
            q_after, c_after = select_after(request, q)
            require(q_after == q and c_after['Executable'] == executable, 'selection changed')
            absent(process)
            absent(inner_process)
            cleanup_confirmed = True
            original = q['Retained']['Producer']['Receipt']
            require(read_pin(original).get('Status') == 'failed', 'original failed receipt changed/status differs')
            os.kill(waiter.pid, signal.SIGCONT)
            stopped = False
            require(waiter.wait(timeout=5) == 0, 'service waiter failed')
            summary = finish(directory, request, q, executable, process, receipt, raw, unit)
            summary.update(Status='OBSERVED_SELECTED', OriginalFailedReceipt=original)
            write(directory / 'summary.json', summary)
    except BaseException as exc:
        failure = str(exc)
    finally:
        if directory is not None:
            # Use only the owned transient unit, never a possibly reused observer PGID.
            if service_owned and failure and not cleanup_confirmed:
                try:
                    command(['/usr/bin/systemctl', '--user', 'stop', unit], directory, 'failure-stop')
                except Exception as exc:
                    failure += '; cleanup: ' + str(exc)
            if service_requested and not cleanup_confirmed:
                try:
                    write(directory / 'cleanup-pending.json', dict(Unit=unit, Error=failure,
                          Awaiting='owned service/process/group/cgroup absence or external watchdog', UTC=utc()))
                except Exception as exc:
                    failure = (failure or '') + '; pending receipt: ' + str(exc)
                # A client exit (including a bus error) says nothing about its
                # detached service. Unknown activation has no safe local deadline.
                # Keep the descriptor open, with no observer/stop retry, until
                # independent absence succeeds or the external watchdog kills us.
                probe = 0
                while not cleanup_confirmed:
                    try:
                        require(service_owned and inner_process is not None, 'service ownership unknown')
                        absent(inner_process)
                        if process is not None:
                            absent(process)
                        raw = command(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'ControlGroup'], directory, 'failure-show-' + str(probe))
                        properties(raw)
                        cleanup_confirmed = True
                    except Exception as exc:
                        if probe == 0:
                            failure = (failure or '') + '; absence: ' + str(exc)
                    probe += 1
                    if not cleanup_confirmed:
                        time.sleep(1)
                write(directory / 'failure-cleanup.json', dict(Unit=unit, Process=process,
                      InnerProcess=inner_process, IndependentlyAbsent=True, RawCommand=raw, UTC=utc()))
            if waiter is not None:
                if stopped and waiter.poll() is None:
                    os.kill(waiter.pid, signal.SIGCONT)
                try:
                    # Reaping/killing the client is safe for exclusion only after
                    # the service is independently absent.
                    waiter.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    waiter.kill()  # This unreaped Popen child cannot be a foreign PID.
                    waiter.wait(timeout=2)
                write(directory / 'service-result.json', dict(Exit=waiter.returncode, UTC=utc(),
                      Stdout=pin(directory / 'service.stdout'), Stderr=pin(directory / 'service.stderr')))
            write(directory / 'outer-result.json', dict(Error=failure, UTC=utc(), Summary=summary))
        if lock is not None:
            os.close(lock)
    if failure:
        print(json.dumps(dict(Error=failure, ReceiptDirectory=str(directory) if directory else None)), file=sys.stderr)
        return 1
    print(json.dumps(summary))
    return 0


def select_after(request, q):
    # Output is now reserved by the observer; retain all other selection checks.
    current = read_pin(request)
    require(current == q, 'request changed after launch')
    controller = read_pin(q['ObserverController'])
    check_pin(controller['Executable'])
    return current, controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request')
    parser.add_argument('--request-sha256')
    parser.add_argument('--receipt-dir')
    parser.add_argument('--_inner', help=argparse.SUPPRESS)
    parser.add_argument('--_inner-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._inner:
        require(not any([args.request, args.request_sha256, args.receipt_dir]) and args._inner_sha256, 'private stage selectors differ')
        return inner({'Path': args._inner, 'SHA256': args._inner_sha256})
    if not all([args.request, args.request_sha256, args.receipt_dir]) or args._inner_sha256:
        parser.error('--request, independent --request-sha256 and --receipt-dir are mandatory')
    return launch({'Path': args.request, 'SHA256': args.request_sha256}, args.receipt_dir)


if __name__ == '__main__':
    sys.exit(main())
