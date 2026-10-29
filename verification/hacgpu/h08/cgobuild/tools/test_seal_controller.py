import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('seal_controller', Path(__file__).with_name('seal_controller.py'))
seal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seal)


class GlueControls(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / 'checkout'
        self.cmd = self.root / seal.TARGET[2:]
        self.cmd.mkdir(parents=True)
        (self.cmd / 'main.go').write_text('package main\nfunc main() {}\n')
        (self.root / 'go.mod').write_text(seal.MODULE + '\n\ngo 1.24.0\n')
        self.extra = self.cmd / 'extra.go'
        self.extra.write_text('package main\n')
        self.go = self.base / 'go'
        names = ['VERSION', 'go.env', 'bin/go', 'pkg/tool/linux_amd64/asm',
                 'pkg/tool/linux_amd64/compile', 'pkg/tool/linux_amd64/link',
                 'pkg/include/asm_amd64.h', 'pkg/include/funcdata.h', 'pkg/include/textflag.h']
        tools = {}
        for name in names:
            p = self.go / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('go1.27.3\n' if name == 'VERSION' else name)
            tools[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.controller = self.base / 'old.json'
        self.controller.write_text(json.dumps({'Tools': tools}))
        self.out = self.root / 'artifacts/tensor/attempt'
        self.args = ['--root', str(self.root), '--tools-controller', str(self.controller),
                     '--tools-controller-sha256', hashlib.sha256(self.controller.read_bytes()).hexdigest(),
                     '--out', str(self.out)]
        self.calls = []
        self.addCleanup(patch.stopall)
        patch.object(seal, 'GO_ROOT', self.go).start()
        patch.object(seal, 'DEADLINE', time.monotonic() + 175).start()
        patch.object(seal, 'effective', return_value=[{'mocked': True}]).start()
        patch.object(seal.subprocess, 'Popen', side_effect=self.launch).start()
        patch.object(seal.os, 'killpg', return_value=None).start()
        patch.object(seal.os, 'waitid', return_value=object(), create=True).start()

    def launch(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[1] == 'list':
            payload = dict(Dir=str(self.cmd), GoFiles=['main.go', 'extra.go'],
                           Module={'GoMod': str(self.root / 'go.mod')})
            kwargs['stdout'].write(json.dumps(payload).encode())
        elif argv[1] == 'build':
            Path(argv[argv.index('-o') + 1]).write_bytes(b'mocked executable')
        elif argv[1:3] == ['-mode', 'check-controller']:
            descriptor = json.loads(Path(argv[4]).read_text())
            if len(descriptor['Snapshots']) != len(descriptor['Inputs']):
                kwargs['stderr'].write(b'sealed CGO0 interposer custody incomplete\n')
                return type('Child', (), {'pid': 12345, 'wait': lambda s, timeout: 1})()
        else:
            raise AssertionError(f'unexpected launch: {argv}')
        return type('Child', (), {'pid': 12345, 'wait': lambda s, timeout: 0})()

    def invoke(self, args=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return seal.main(self.args if args is None else args)

    def test_missing_selectors_never_launch(self):
        for option in ['--root', '--tools-controller', '--tools-controller-sha256', '--out']:
            args = list(self.args)
            i = args.index(option)
            del args[i:i + 2]
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                seal.main(args)
        self.assertEqual(self.calls, [])

    def test_invalid_root_source_and_output_never_launch(self):
        wrong = list(self.args)
        wrong[1] = str(self.base)
        self.assertEqual(self.invoke(wrong), 1)
        (self.cmd / 'main.go').unlink()
        self.assertEqual(self.invoke(), 1)
        (self.cmd / 'main.go').write_text('package main\n')
        (self.root / 'go.mod').write_text('module wrong\n')
        self.assertEqual(self.invoke(), 1)
        (self.root / 'go.mod').write_text(seal.MODULE + '\n')
        outside = list(self.args)
        outside[-1] = str(self.base / 'outside')
        self.assertEqual(self.invoke(outside), 1)
        self.out.mkdir(parents=True)
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.calls, [])

    def test_wrong_independent_digest_never_launches(self):
        args = list(self.args)
        args[5] = '0' * 64
        self.assertEqual(self.invoke(args), 1)
        self.assertEqual(self.calls, [])

    def test_effective_limit_refusal_never_launches(self):
        with patch.object(seal, 'effective', side_effect=ValueError('unbounded')):
            self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.calls, [])

    def test_source_changes_before_build_refuse_child(self):
        original = Path.mkdir
        def changed(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.name == 'build' and path.parent.name == 'commands':
                self.extra.write_text('package main\n// changed after snapshot\n')
            return result
        with patch.object(Path, 'mkdir', changed):
            self.assertEqual(self.invoke(), 1)
        self.assertEqual([a[1] for a in self.calls], ['list'])
        failed = json.loads((self.out / 'commands/build/command.json').read_text())
        self.assertEqual(failed['exit_code'], -1)
        self.assertIn('input changed before child', (self.out / 'failure.json').read_text())

    def test_tool_changes_before_build_refuse_child(self):
        original = Path.mkdir
        def changed(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.name == 'build' and path.parent.name == 'commands':
                (self.go / 'bin/go').write_text('substituted tool')
            return result
        with patch.object(Path, 'mkdir', changed):
            self.assertEqual(self.invoke(), 1)
        self.assertEqual([a[1] for a in self.calls], ['list'])

    def test_mocked_seal_launches_only_fixed_commands_and_rehashed_control(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual([a[1] for a in self.calls], ['list', 'build', 'list', '-mode', '-mode'])
        build = self.calls[1]
        self.assertEqual(build[1:], ['build', '-a', '-x', '-work', '-p=2', '-pgo=off',
                                    '-trimpath', '-buildvcs=false', '-o', str(self.out / 'cgo-plan'), seal.TARGET])
        last = self.calls[-1]
        self.assertEqual(last[-1], hashlib.sha256(Path(last[-3]).read_bytes()).hexdigest())
        self.assertNotEqual(self.calls[-2][-1], last[-1])
        for command in (self.out / 'commands').glob('*/command.json'):
            receipt = json.loads(command.read_text())
            self.assertIn('CGO_ENABLED=0', receipt['environment_overrides'])
            self.assertIn('GOPROXY=off', receipt['environment_overrides'])
            self.assertTrue(receipt['stdout_sha256'])


class OwnedChildControls(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'commands').mkdir()
        self.events = []
        self.reaped, self.exited = False, True
        self.exit_code, self.now = 0, 100.0
        self.child = Mock(pid=12345)
        self.child.wait.side_effect = self.reap
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        for target, name, kwargs in [
                (seal, 'authenticate', dict(return_value=None)),
                (seal, 'effective', dict(return_value=[{'mocked': True}])),
                (seal, 'utc', dict(return_value='mocked UTC')),
                (seal, 'DEADLINE', dict(new=275.0)),
                (seal.time, 'monotonic', dict(side_effect=lambda: self.now)),
                (seal.time, 'sleep', dict(side_effect=self.sleep)),
                (seal.subprocess, 'Popen', dict(side_effect=self.launch)),
                (seal.os, 'waitid', dict(side_effect=self.observe, create=True)),
                (seal.os, 'killpg', dict(side_effect=self.kill))]:
            self.patches.enter_context(patch.object(target, name, **kwargs))
        self.runner = seal.Seal(self.root, self.root, {'MOCKED': '1'}, {})

    def launch(self, argv, **kwargs):
        self.assertTrue(kwargs['start_new_session'])
        kwargs['stdout'].write(b'mocked stdout\n')
        kwargs['stderr'].write(b'mocked stderr\n')
        return self.child

    def observe(self, kind, pid, flags):
        self.assertEqual((kind, pid, flags),
                         (seal.os.P_PID, self.child.pid,
                          seal.os.WEXITED | seal.os.WNOHANG | seal.os.WNOWAIT))
        self.events.append('observe')
        if self.reaped:
            raise ChildProcessError('mocked child identity no longer owned')
        return object() if self.exited else None

    def sleep(self, seconds):
        self.assertGreater(seconds, 0)
        self.assertLessEqual(seconds, 0.01)
        self.now = seal.DEADLINE

    def kill(self, pid, sig):
        self.assertFalse(self.reaped,
                         'mocked cleanup signaled a released process-group identity')
        self.assertEqual((pid, sig), (self.child.pid, seal.signal.SIGKILL))
        self.events.append('kill')
        self.exited = True

    def reap(self, timeout):
        self.assertFalse(self.reaped, 'mocked child reaped twice')
        self.assertGreater(timeout, 0)
        self.events.append('reap')
        self.reaped = True
        return self.exit_code

    def run_child(self, expected_exit=0):
        return self.runner.run('mocked', ['mocked-command'], expected_exit=expected_exit)

    def assert_cleanup(self):
        self.assertEqual(self.events.count('kill'), 1)
        self.assertEqual(self.events.count('reap'), 1)
        self.assertLess(self.events.index('kill'), self.events.index('reap'))
        self.child.wait.assert_called_once()

    def assert_receipt(self, exit_code, error=None):
        directory = self.root / 'commands/mocked'
        receipt = json.loads((directory / 'command.json').read_text())
        attempt = json.loads((directory / 'attempt.json').read_text())
        self.assertEqual(receipt['exit_code'], exit_code)
        self.assertEqual(attempt['error'], error)
        self.assertEqual(attempt['process'], {'PID': 12345, 'PGID': 12345})
        self.assertEqual(receipt['stdout_sha256'], hashlib.sha256(b'mocked stdout\n').hexdigest())
        self.assertEqual(receipt['stderr_sha256'], hashlib.sha256(b'mocked stderr\n').hexdigest())

    def test_mocked_normal_exit_never_signals_after_reap(self):
        self.assertEqual(self.run_child()['exit_code'], 0)
        self.assert_cleanup()
        self.assert_receipt(0)

    def test_mocked_expected_exit_one_never_signals_after_reap(self):
        self.exit_code = 1
        self.assertEqual(self.run_child(expected_exit=1)['exit_code'], 1)
        self.assert_cleanup()
        self.assert_receipt(1)

    def test_mocked_timeout_cleans_before_single_reap(self):
        self.exited, self.exit_code = False, -seal.signal.SIGKILL
        with self.assertRaisesRegex(TimeoutError, 'shared 175-second deadline exceeded'):
            self.run_child()
        self.assert_cleanup()
        self.assert_receipt(self.exit_code, 'shared 175-second deadline exceeded')

    def test_mocked_interruption_cleans_before_single_reap(self):
        self.exited, self.exit_code = False, -seal.signal.SIGKILL
        with patch.object(seal.time, 'sleep', side_effect=KeyboardInterrupt('mocked interruption')):
            with self.assertRaisesRegex(KeyboardInterrupt, 'mocked interruption'):
                self.run_child()
        self.assert_cleanup()
        self.assert_receipt(self.exit_code, 'mocked interruption')

    def test_mocked_unexpected_exit_keeps_failure_and_cleanup(self):
        self.exit_code = 2
        with self.assertRaisesRegex(ValueError, 'mocked: exit 2, expected 0'):
            self.run_child()
        self.assert_cleanup()
        self.assert_receipt(2)

    def test_mocked_lost_ownership_refuses_signal_and_reap(self):
        self.reaped = True
        with self.assertRaisesRegex(ChildProcessError, 'identity no longer owned'):
            self.run_child()
        self.assertNotIn('kill', self.events)
        self.child.wait.assert_not_called()
        self.assert_receipt(-1, 'mocked child identity no longer owned')

    def test_mocked_cleanup_ownership_check_failure_refuses_signal_and_reap(self):
        error = ChildProcessError('mocked cleanup ownership lost')
        with patch.object(seal.os, 'waitid', side_effect=[object(), error]):
            with self.assertRaisesRegex(ChildProcessError, 'cleanup ownership lost'):
                self.run_child()
        self.assertNotIn('kill', self.events)
        self.child.wait.assert_not_called()
        self.assert_receipt(-1, 'mocked cleanup ownership lost')

    def test_mocked_process_receipt_failure_still_cleans_owned_child(self):
        original = seal.write

        def fail_process_receipt(path, value):
            if path.name == 'process.json':
                raise OSError('mocked process receipt failure')
            return original(path, value)

        with patch.object(seal, 'write', side_effect=fail_process_receipt):
            with self.assertRaisesRegex(OSError, 'mocked process receipt failure'):
                self.run_child()
        self.assert_cleanup()
        self.assert_receipt(0, 'mocked process receipt failure')

    def test_mocked_signal_failure_is_not_accepted_as_clean(self):
        with patch.object(seal.os, 'killpg', side_effect=PermissionError('mocked kill refused')):
            with self.assertRaisesRegex(PermissionError, 'mocked kill refused'):
                self.run_child()
        self.child.wait.assert_called_once()
        self.assert_receipt(0, 'mocked kill refused')

    def test_mocked_cleanup_wait_timeout_does_not_retry_or_signal_again(self):
        self.child.wait.side_effect = seal.subprocess.TimeoutExpired('mocked-command', 175)
        with self.assertRaisesRegex(TimeoutError, 'cleanup deadline exceeded'):
            self.run_child()
        self.assertEqual(self.events.count('kill'), 1)
        self.child.wait.assert_called_once()
        self.assert_receipt(-1, 'cleanup deadline exceeded')

    def test_mocked_reap_interruption_never_signals_or_reaps_again(self):
        def interrupted_reap(timeout):
            self.reap(timeout)
            raise KeyboardInterrupt('mocked reap interruption')

        self.child.wait.side_effect = interrupted_reap
        with self.assertRaisesRegex(KeyboardInterrupt, 'mocked reap interruption'):
            self.run_child()
        self.assert_cleanup()
        self.assert_receipt(-1, 'mocked reap interruption')


if __name__ == '__main__':
    unittest.main()
