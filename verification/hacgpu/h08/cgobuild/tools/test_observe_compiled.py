"""Synthetic mocked boundaries only. These records are NOT execution witnesses."""
import importlib.util
import json
from pathlib import Path
import tempfile
from contextlib import ExitStack
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('observe', Path(__file__).with_name('observe_compiled.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SyntheticControls(unittest.TestCase):
    def setUp(self):
        self.clock = 0.0
        patch.object(m.time, 'monotonic', side_effect=lambda: self.clock).start()
        patch.object(m.time, 'sleep', side_effect=lambda seconds: self.advance(seconds)).start()
        patch.object(m, 'utc', return_value='2026-10-28T00:00:04+00:00').start()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / 'artifacts/tensor'
        self.base.mkdir(parents=True)
        self.output = self.base / 'output'
        self.directory = self.base / 'receipt'
        self.exe = self.root / 'observer'
        self.exe.write_bytes(b'SYNTHETIC - NEVER EXECUTE')
        self.executable = m.pin(self.exe)
        self.controller = dict(Schema='nanopore-h08-cgo-interposer/v1', Executable=self.executable,
            Inputs={str(self.root / 'verification/hacgpu/h08/cgobuild' / n): 'a' * 64
                    for n in ['observation.go', 'cmd/main.go']}, Snapshots={}, Tools={}, Lists=[], Build={})
        self.cp = m.write(self.root / 'controller.json', self.controller)
        fake = {'Path': str(self.root / 'unused'), 'SHA256': 'a' * 64}
        self.q = dict(Schema='nanopore-h08-observation-request/v1', Role='h08-command',
            Retained=dict(Producer={n: fake for n in ['Receipt', 'Plan', 'Controller', 'Queries', 'Output']},
                          Handoff=fake, Outer=fake, Cleanup=fake, FailureStderr=fake,
                          Profile='saved005-final-reader-v1'), ObserverController=self.cp,
            ObserverRoot=str(self.root), AlgorithmReferences=m.ALGORITHMS, OutputDirectory=str(self.output))
        self.request = m.write(self.root / 'request.json', self.q)
        self.process = dict(PID=123, Start='123456', Cgroup='0::/synthetic.service\n', Group=True)
        for obj, names in [(m.subprocess, ['Popen', 'run']),
                           (m.os, ['open', 'close', 'fstat', 'kill', 'killpg']),
                           (m.fcntl, ['flock'])]:
            for name in names:
                patch.object(obj, name, side_effect=AssertionError('unmocked system boundary: ' + name)).start()

    def advance(self, seconds):
        self.clock += seconds

    def test_missing_selectors(self):
        for args in [[], ['--request', str(self.root / 'request.json')],
                     ['--request', str(self.root / 'request.json'), '--receipt-dir', str(self.directory)]]:
            with patch.object(m, 'launch') as launch, self.assertRaises(SystemExit):
                m.main(args)
            launch.assert_not_called()

    def test_wrong_independent_selector_and_rehashed_controller(self):
        with self.assertRaisesRegex(ValueError, 'digest'):
            m.select(dict(self.request, SHA256='0' * 64))
        changed = dict(self.controller, Executable=dict(self.executable, SHA256='0' * 64))
        cp = m.write(self.root / 'changed-controller.json', changed)
        rp = m.write(self.root / 'changed-request.json', dict(self.q, ObserverController=cp))
        with self.assertRaisesRegex(ValueError, 'digest'):
            m.select(rp)

    def test_preexisting_output_and_receipt(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'reuse'):
            m.select(self.request)
        self.directory.mkdir()
        with self.assertRaisesRegex(ValueError, 'reuse'):
            m.fresh(self.directory, self.root)

    def test_foreign_live_executable_and_argv(self):
        for exe, args in [('foreign', b'wrong\0'), (self.executable['Path'], b'wrong\0')]:
            with patch.object(m, 'identity', return_value=self.process), \
                 patch.object(m.os, 'readlink', return_value=exe), \
                 patch.object(m, 'digest', return_value=self.executable['SHA256']), \
                 patch.object(Path, 'read_bytes', return_value=args), \
                 self.assertRaisesRegex(ValueError, 'foreign'):
                m.verify_live(self.process, self.executable, ['expected'], self.root, 'synthetic.service')

    def records(self):
        self.directory.mkdir()
        self.output.mkdir()
        observation = m.write(self.output / 'observation.json', {'Synthetic': True})
        stdout = self.directory / 'observer.stdout'
        stderr = self.directory / 'observer.stderr'
        stdout.write_text(json.dumps(observation) + '\n')
        stderr.write_bytes(b'')
        r = dict(command=m.observer_args(self.executable, self.request, str(self.output)),
                 environment_overrides=m.ENV, started_utc='2026-10-28T00:00:00+00:00',
                 finished_utc='2026-10-28T00:00:01+00:00', exit_code=0,
                 stdout=str(stdout), stderr=str(stderr), stdout_sha256=m.digest(stdout), stderr_sha256=m.digest(stderr))
        a, b = self.directory / 'cleanup.stdout', self.directory / 'cleanup.stderr'
        a.write_text('ActiveState=inactive\nSubState=dead\nMainPID=0\nControlGroup=\n')
        b.write_bytes(b'')
        raw = dict(command=['systemctl', '--user', 'show', 'synthetic.service', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'ControlGroup'],
                   environment_overrides=[], started_utc='2026-10-28T00:00:02+00:00',
                   finished_utc='2026-10-28T00:00:03+00:00', exit_code=0,
                   stdout=str(a), stderr=str(b), stdout_sha256=m.digest(a), stderr_sha256=m.digest(b))
        return r, raw

    def finish(self, r, raw):
        return m.finish(self.directory, self.request, self.q, self.executable,
                        self.process, r, raw, 'synthetic.service')

    def test_synthetic_positive_exact_wire_and_order(self):
        r, raw = self.records()
        writes = []
        real_write = m.write
        def record(path, value):
            writes.append(Path(path).name)
            return real_write(path, value)
        with patch.object(m, 'write', side_effect=record), patch.object(m.subprocess, 'Popen') as spawn:
            result = self.finish(r, raw)
        spawn.assert_not_called()
        self.assertEqual(writes, ['cleanup.json', 'execution.json', 'selection.json'])
        x = m.read_pin(result['Execution'])
        self.assertEqual(set(x), set('Schema Request Observation ObserverController Executable Process Cwd Command Cleanup'.split()))
        self.assertEqual(x['Command']['environment_overrides'], m.ENV)
        self.assertEqual(set(m.read_pin(result['Cleanup'])), set('Schema UTC Unit Process PIDAbsent GroupAbsent CgroupAbsent RawCommand'.split()))
        self.assertEqual(set(m.read_pin(result['Selection'])), {'Schema', 'Request', 'Execution'})

    def test_failed_child_cleanup_and_rehashed_output_reject(self):
        r, raw = self.records()
        for bad_r, bad_raw in [(dict(r, exit_code=1), raw),
                               (dict(r, command=['foreign']), raw),
                               (r, dict(raw, exit_code=1))]:
            with self.assertRaises(ValueError):
                self.finish(bad_r, bad_raw)
            self.assertFalse((self.directory / 'selection.json').exists())
        observation = json.loads(Path(r['stdout']).read_text())
        Path(observation['Path']).chmod(0o600)
        Path(observation['Path']).write_bytes(b'corrupted')
        # Rehashing the outer stdout does not repair the independently emitted pin.
        r['stdout_sha256'] = m.digest(r['stdout'])
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.finish(r, raw)
        self.assertFalse((self.directory / 'selection.json').exists())

    def launcher(self, name, *, stop_error=None, fail_live=False, early_exit=False,
                 owned=True, completed=True, death_at=None, probe_error_until=0,
                 show_error_until=0, watchdog=False, ready_at=0, handshake_delay=0,
                 client_timeout=False):
        original = m.write(self.root / 'failed.json', {'Status': 'failed'})
        self.q['Retained']['Producer']['Receipt'] = original
        request = m.write(self.root / 'full-request.json', self.q)
        receipt_dir = self.base / name
        state = dict(live=True, held=False, proof=False, watchdog=False, armed=False,
                     events=[], signals=[], stop_calls=0, kill_calls=0, polls=0)
        actual_resolve = Path.resolve
        actual_exists, real_pin, real_write = Path.exists, m.pin, m.write
        def resolve(path, *args, **kwargs):
            return actual_resolve(path, *args, **kwargs) if path.is_relative_to(self.root) else path
        def refresh():
            if death_at is not None and self.clock >= death_at:
                state['live'] = False
        def exists(path):
            refresh()
            if path.is_relative_to('/sys/fs/cgroup'):
                return state['live']
            if path == receipt_dir / 'inner-ready.json' and self.clock < ready_at:
                return False
            if path == receipt_dir / 'spawn.json' and not state['armed']:
                return False
            return actual_exists(path)
        def runtime_pin(path):
            p = Path(path)
            if p.is_relative_to(self.root):
                return real_pin(p)
            return dict(Path=str(p), SHA256='a' * 64)
        def write(path, value):
            if Path(path).name == 'arm':
                state['armed'] = True
                state['events'].append(('arm', self.clock))
            if Path(path).name == 'release' and death_at is None:
                state['live'] = False
            return real_write(path, value)
        def open_lock(path, flags):
            self.assertEqual(path, m.LOCK)
            self.assertEqual(flags, m.os.O_RDONLY | m.os.O_CLOEXEC | m.os.O_NOFOLLOW)
            state['held'] = True
            return 1234
        def close_lock(fd):
            refresh()
            self.assertEqual(fd, 1234)
            self.assertFalse(state['live'] and not state['watchdog'],
                             'host lock closed while detached service remains live')
            self.assertTrue(state['proof'] or state['watchdog'],
                            'host lock closed without absence proof or watchdog termination')
            state['held'] = False
            state['events'].append(('close', self.clock))
        def sleep(seconds):
            self.assertTrue(state['held'], 'cleanup progressed without host exclusion')
            self.advance(seconds)
            refresh()
            if watchdog and self.clock >= 600:
                state['watchdog'] = True
                state['events'].append(('watchdog', self.clock))
                raise WatchdogTermination()
            self.assertLess(self.clock, 600, 'mock cleanup exceeded its external watchdog budget')
        class WatchdogTermination(BaseException):
            pass
        test = self
        class Waiter:
            pid = 456789
            returncode = None
            def poll(self):
                state['polls'] += 1
                if early_exit:
                    self.returncode = 17
                return self.returncode
            def wait(self, timeout):
                refresh()
                state['events'].append(('wait', test.clock))
                if early_exit:
                    self.returncode = 17
                elif self.returncode is None:
                    if state['live'] or client_timeout:
                        sleep(timeout)
                        raise m.subprocess.TimeoutExpired('synthetic waiter', timeout)
                    self.returncode = 0
                return self.returncode
            def kill(self):
                state['kill_calls'] += 1
                state['events'].append(('kill-waiter', test.clock))
                self.returncode = -9
        def spawn(argv, **kwargs):
            unit = argv[4].split('=', 1)[1]
            process = dict(self.process, Cgroup='0::/' + unit + '\n')
            state.update(unit=unit, process=process)
            self.assertTrue(state['held'])
            self.assertEqual(argv[5:14], ['--property=RuntimeMaxSec=575s',
                '--property=TimeoutStartSec=5s', '--property=TimeoutStopSec=5s',
                '--property=KillMode=control-group', '--property=SendSIGKILL=yes',
                '--property=CPUQuota=200%', '--property=MemoryMax=3G',
                '--property=MemorySwapMax=0', '--property=TasksMax=64'])
            if owned:
                m.write(receipt_dir / 'inner-ready.json', dict(process, PID=456))
                m.write(receipt_dir / 'spawn.json', process)
            if completed:
                self.output.mkdir()
                observation = m.write(self.output / 'observation.json', {'Synthetic': True})
                out, err = receipt_dir / 'observer.stdout', receipt_dir / 'observer.stderr'
                out.write_text(json.dumps(observation) + '\n')
                err.write_bytes(b'')
                receipt = dict(command=m.observer_args(self.executable, request, str(self.output)),
                    environment_overrides=m.ENV, started_utc='2026-10-28T00:00:00+00:00',
                    finished_utc='2026-10-28T00:00:01+00:00', exit_code=0,
                    stdout=str(out), stderr=str(err), stdout_sha256=m.digest(out), stderr_sha256=m.digest(err))
                m.write(receipt_dir / 'observer.command.json', receipt)
                m.write(receipt_dir / 'inner-result.json', {'Error': None})
            return Waiter()
        def absent(process):
            refresh()
            self.assertTrue(state['held'])
            if state['live']:
                raise ValueError('owned cgroup remains')
            if self.clock < probe_error_until:
                raise OSError('synthetic absence probe unavailable')
            state['events'].append(('absent-' + str(process['PID']), self.clock))
        def command(argv, directory, label, timeout=8):
            self.assertTrue(state['held'])
            if label == 'failure-stop':
                state['stop_calls'] += 1
                if stop_error == 'timeout':
                    raise m.subprocess.TimeoutExpired(argv, timeout)
                if stop_error == 'nonzero':
                    raise ValueError('failure-stop command failed')
                if death_at is None:
                    state['live'] = False
            else:
                if self.clock < show_error_until:
                    raise ValueError('synthetic system bus error')
                self.assertFalse(state['live'])
                state['proof'] = True
                state['events'].append(('service-absent', self.clock))
            a, b = directory / (label + '.stdout'), directory / (label + '.stderr')
            a.write_text('ActiveState=inactive\nSubState=dead\nMainPID=0\nControlGroup=\n')
            b.write_bytes(b'')
            return dict(command=argv, environment_overrides=[],
                started_utc='2026-10-28T00:00:02+00:00', finished_utc='2026-10-28T00:00:03+00:00',
                exit_code=0, stdout=str(a), stderr=str(b), stdout_sha256=m.digest(a), stderr_sha256=m.digest(b))
        def verify_inner(*args):
            self.advance(handshake_delay)
        def verify_live(*args):
            if fail_live:
                raise ValueError('synthetic live inspection failure')
        def send(pid, sig):
            self.assertEqual(pid, Waiter.pid, 'signal targeted a foreign numeric identity')
            self.assertFalse(early_exit, 'signal targeted an exited/reaped waiter')
            self.assertIn(sig, [m.signal.SIGSTOP, m.signal.SIGCONT])
            state['signals'].append(sig)
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, 'resolve', resolve))
            stack.enter_context(patch.object(Path, 'exists', exists))
            flock = stack.enter_context(patch.object(m.fcntl, 'flock'))
            popen = stack.enter_context(patch.object(m.subprocess, 'Popen', side_effect=spawn))
            for obj, attr, kwargs in [
                (m, 'effective', dict(return_value=[])),
                (m, 'select', dict(return_value=(self.q, self.controller))),
                (m, 'select_after', dict(return_value=(self.q, self.controller))),
                (m, 'pin', dict(side_effect=runtime_pin)),
                (m, 'write', dict(side_effect=write)),
                (m.os, 'open', dict(side_effect=open_lock)),
                (m.os, 'fstat', dict(return_value=type('Stat', (), dict(st_dev=1, st_ino=2))())),
                (m.os, 'close', dict(side_effect=close_lock)),
                (m.os, 'kill', dict(side_effect=send)),
                (m.os, 'killpg', dict(side_effect=AssertionError('numeric group signal forbidden'))),
                (m, 'verify_inner', dict(side_effect=verify_inner)),
                (m, 'verify_live', dict(side_effect=verify_live)),
                (m, 'absent', dict(side_effect=absent)),
                (m, 'command', dict(side_effect=command)),
                (m.subprocess, 'run', dict(side_effect=AssertionError('real command forbidden'))),
                (m.time, 'sleep', dict(side_effect=sleep))]:
                stack.enter_context(patch.object(obj, attr, **kwargs))
            try:
                state['status'] = m.launch(request, str(receipt_dir))
            except WatchdogTermination:
                self.assertTrue(watchdog)
                self.assertTrue(state['held'], 'descriptor must remain open until watchdog termination')
                self.assertFalse((receipt_dir / 'selection.json').exists())
                state['status'] = 'watchdog'
                close_lock(1234)
            flock.assert_called_once_with(1234, m.fcntl.LOCK_EX | m.fcntl.LOCK_NB)
            popen.assert_called_once()
        self.assertEqual(m.read_pin(original), {'Status': 'failed'})
        self.assertFalse(state['held'])
        state['directory'] = receipt_dir
        state['request'] = request
        return state

    def test_synthetic_complete_launcher_and_cleanup_failure(self):
        state = self.launcher('full-positive')
        self.assertEqual(state['status'], 0)
        selection = m.read_pin(m.pin(state['directory'] / 'selection.json'))
        self.assertEqual(set(selection), {'Schema', 'Request', 'Execution'})
        self.assertEqual(selection['Request'], state['request'])
        self.assertEqual(state['signals'], [m.signal.SIGSTOP, m.signal.SIGCONT])
        self.assertEqual(state['stop_calls'], 0)

    def test_failed_stop_still_running_waiter_retains_lock(self):
        state = self.launcher('failed-stop-running', fail_live=True,
                              stop_error='timeout', death_at=30)
        self.assertEqual(state['status'], 1)
        self.assertFalse((state['directory'] / 'selection.json').exists())
        self.assertEqual(state['stop_calls'], 1)
        self.assertEqual(state['kill_calls'], 0)
        self.assertGreaterEqual(dict(state['events'])['close'], 30)
        self.assertTrue((state['directory'] / 'failure-cleanup.json').exists())

    def test_failed_stop_nonzero_and_early_waiter_exit_never_proves_absence(self):
        state = self.launcher('failed-stop-early', early_exit=True, completed=False,
                              stop_error='nonzero', death_at=40)
        self.assertEqual(state['status'], 1)
        self.assertFalse((state['directory'] / 'selection.json').exists())
        self.assertEqual(state['signals'], [])
        self.assertEqual(state['stop_calls'], 1)
        self.assertGreaterEqual(dict(state['events'])['close'], 40)

    def test_unknown_startup_client_exit_holds_until_external_watchdog(self):
        state = self.launcher('unknown-bus-error', early_exit=True, owned=False,
                              completed=False, watchdog=True)
        self.assertEqual(state['status'], 'watchdog')
        self.assertEqual(state['signals'], [])
        self.assertEqual(state['stop_calls'], 0)
        self.assertFalse(state['armed'])
        self.assertTrue(state['live'])
        self.assertEqual([event[0] for event in state['events']][-2:], ['watchdog', 'close'])

    def test_absence_probe_error_retains_lock_and_resumes_stopped_waiter(self):
        state = self.launcher('absence-probe-error', probe_error_until=12)
        self.assertEqual(state['status'], 1)
        self.assertFalse((state['directory'] / 'selection.json').exists())
        self.assertEqual(state['signals'], [m.signal.SIGSTOP, m.signal.SIGCONT])
        self.assertGreaterEqual(dict(state['events'])['close'], 12)

    def test_service_show_bus_error_retains_lock_despite_process_absence(self):
        state = self.launcher('show-bus-error', show_error_until=15)
        self.assertEqual(state['status'], 1)
        self.assertFalse((state['directory'] / 'selection.json').exists())
        self.assertGreaterEqual(dict(state['events'])['close'], 15)

    def test_unresolved_known_service_holds_until_external_watchdog(self):
        state = self.launcher('known-unresolved', fail_live=True, stop_error='timeout',
                              completed=False, watchdog=True)
        self.assertEqual(state['status'], 'watchdog')
        self.assertEqual(state['kill_calls'], 0)
        self.assertEqual(state['stop_calls'], 1)
        self.assertTrue(state['armed'])
        self.assertTrue(state['live'])

    def test_proven_absence_allows_owned_waiter_timeout_kill(self):
        state = self.launcher('client-timeout', fail_live=True, client_timeout=True)
        self.assertEqual(state['status'], 1)
        self.assertEqual(state['kill_calls'], 1)
        self.assertFalse((state['directory'] / 'selection.json').exists())
        names = [event[0] for event in state['events']]
        self.assertLess(names.index('service-absent'), names.index('kill-waiter'))
        self.assertLess(names.index('kill-waiter'), names.index('close'))

    def test_parent_late_handshake_cannot_arm(self):
        state = self.launcher('late-handshake', completed=False, ready_at=11,
                              watchdog=True)
        self.assertEqual(state['status'], 'watchdog')
        self.assertFalse(state['armed'])

    def test_parent_slow_identity_check_cannot_arm(self):
        state = self.launcher('slow-handshake', completed=False, handshake_delay=11)
        self.assertEqual(state['status'], 1)
        self.assertFalse(state['armed'])
        self.assertFalse((state['directory'] / 'selection.json').exists())

    def test_inner_late_selection_cannot_start_go(self):
        self.directory.mkdir()
        (self.directory / 'arm').write_text('{}')
        (self.directory / 'release').write_text('{}')
        config = dict(Directory=str(self.directory), Request=self.request,
                      Runtime={}, Executable=self.executable, StartupDeadline=10)
        def select(request):
            self.advance(11)
            return self.q, self.controller
        with patch.object(m, 'read_pin', return_value=config), \
             patch.object(m, 'effective', return_value=[]), \
             patch.object(m, 'identity', return_value=self.process), \
             patch.object(m.os, 'getpid', return_value=123), \
             patch.object(m, 'select', side_effect=select), \
             patch.object(m.subprocess, 'Popen') as spawn, \
             patch.object(m.os, 'kill') as send:
            self.assertEqual(m.inner({'Path': 'synthetic', 'SHA256': 'a' * 64}), 1)
        spawn.assert_not_called()
        send.assert_not_called()
        self.assertIn('startup budget', json.loads((self.directory / 'inner-result.json').read_text())['Error'])

    def test_cleanup_identity_still_present_rejects(self):
        with patch.object(m, 'identity', return_value=self.process), \
             self.assertRaisesRegex(ValueError, 'remains'):
            m.absent(self.process)

    def test_launch_refusal_never_starts_service_or_acquires_lock(self):
        with patch.object(m, 'effective', return_value=[]), \
             patch.object(m.subprocess, 'Popen') as spawn, patch.object(m.fcntl, 'flock') as flock:
            self.assertEqual(m.launch(dict(self.request, SHA256='0' * 64), str(self.directory)), 1)
        spawn.assert_not_called()
        flock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
