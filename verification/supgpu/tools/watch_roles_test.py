"""Cheap process-identity controls; no native loader or real signal is used."""

import copy
import importlib.util
import json
import os
import pathlib
import signal
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'role_watch', pathlib.Path(__file__).with_name('watch_roles.py'))
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def identity(pid=200, parent=100, birth=10):
    return {'pid': pid, 'state': 'S', 'ppid': parent, 'pgid': 200,
            'sid': 100, 'start_ticks': birth, 'raw': 'mock stat',
            'image': '/fake/role', 'image_sha256': 'hash',
            'argv': ['/fake/role', 'fixed'], 'cgroup': '0::/own/test\n'}


class OwnershipControls(unittest.TestCase):
    def setUp(self):
        self.current = {200: identity()}
        self.sent = []
        self.opened = []
        self.events = []
        self.members = [200]
        self.patches = [
            patch.object(watch, 'process_identity', side_effect=self.read_identity),
            patch.object(watch, 'group_members', side_effect=lambda pgid:
                         [copy.deepcopy(self.current[p]) for p in self.members]),
            patch.object(watch.os, 'pidfd_open', side_effect=self.open_pidfd),
            patch.object(watch.os, 'close'),
            patch.object(watch.signal, 'pidfd_send_signal',
                         side_effect=lambda fd, sig: self.sent.append((fd - 1000, sig))),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.group = watch.OwnedGroup(copy.deepcopy(self.current[200]), self.events)

    def tearDown(self):
        evidence = os.environ.get('SUPGPU_WATCHER_TEST_EVIDENCE')
        if evidence:
            path = pathlib.Path(evidence)
            path.mkdir(parents=True, exist_ok=True)
            (path / (self._testMethodName + '.json')).write_text(json.dumps({
                'case': self._testMethodName, 'current': self.current,
                'opened_pidfds': self.opened, 'mocked_signals': self.sent,
                'signal_evidence': self.events, 'real_signals': False,
            }, indent=2) + '\n')

    def read_identity(self, pid, with_hash=False):
        return copy.deepcopy(self.current.get(pid))

    def open_pidfd(self, pid, flags=0):
        self.opened.append(pid)
        return pid + 1000

    def test_reused_leader_is_retired_without_signal(self):
        self.current[200] = identity(parent=999, birth=9999)
        with self.assertRaisesRegex(watch.OwnershipError, 'changed'):
            self.group.signal(signal.SIGTERM)
        self.assertTrue(self.group.retired)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.opened, [])

    def test_gone_leader_signals_only_previously_authenticated_descendant(self):
        self.current[201] = identity(pid=201, parent=200, birth=11)
        self.members = [200, 201]
        self.group.refresh()
        del self.current[200]
        self.current[201]['ppid'] = 1
        self.members = [201]
        self.group.signal(signal.SIGTERM)
        self.assertEqual(self.sent, [(201, signal.SIGTERM)])

    def test_gone_leader_cannot_authorize_new_member(self):
        del self.current[200]
        self.current[201] = identity(pid=201, parent=1, birth=11)
        self.members = [201]
        with self.assertRaisesRegex(watch.OwnershipError, 'unproven'):
            self.group.signal(signal.SIGKILL)
        self.assertEqual(self.sent, [])

    def test_changed_member_identity_rejects_entire_signal_batch(self):
        self.current[201] = identity(pid=201, parent=200, birth=11)
        self.members = [200, 201]
        self.group.refresh()
        self.current[201]['argv'] = ['/fake/unrelated']
        with self.assertRaisesRegex(watch.OwnershipError, 'changed'):
            self.group.signal(signal.SIGKILL)
        self.assertEqual(self.sent, [])

    def test_live_owned_timeout_signal_uses_pidfd(self):
        self.group.signal(signal.SIGTERM)
        self.assertEqual(self.opened, [200])
        self.assertEqual(self.sent, [(200, signal.SIGTERM)])
        self.assertEqual(self.events[0]['identity']['start_ticks'], 10)
        self.assertTrue(self.events[0]['pidfd'])

    def test_completed_group_stays_retired_during_stale_cleanup(self):
        del self.current[200]
        self.members = []
        self.assertEqual(self.group.refresh(), [])
        self.assertTrue(self.group.retired)
        self.current[200] = identity(parent=999, birth=9999)
        self.members = [200]
        self.group.signal(signal.SIGKILL)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.opened, [])

    def test_session_cgroup_and_image_drift_reject_before_signal(self):
        for field, value in [('sid', 999), ('cgroup', '0::/other\n'),
                             ('image', '/fake/unrelated'), ('image_sha256', 'other')]:
            with self.subTest(field=field):
                self.current[200] = identity()
                self.current[200][field] = value
                group = watch.OwnedGroup(identity(), self.events)
                with self.assertRaisesRegex(watch.OwnershipError, 'changed'):
                    group.signal(signal.SIGTERM)
                self.assertEqual(self.sent, [])

    def test_pidfd_open_then_birth_change_never_signals_replacement(self):
        def replace_after_open(pid, flags=0):
            self.current[pid] = identity(parent=999, birth=9999)
            self.opened.append(pid)
            return pid + 1000
        with patch.object(watch.os, 'pidfd_open', side_effect=replace_after_open):
            with self.assertRaisesRegex(watch.OwnershipError, 'changed'):
                self.group.signal(signal.SIGTERM)
        self.assertEqual(self.opened, [200])
        self.assertEqual(self.sent, [])


class RunControls(unittest.TestCase):
    def test_reused_leader_is_not_signaled_by_timeout_or_final_cleanup(self):
        controller = identity(pid=100, parent=50, birth=5)
        controller.update(pgid=100, image='/fake/controller',
                          argv=['/fake/controller'], image_sha256='hash')
        original = identity()
        replacement = identity(parent=999, birth=9999)
        current = {100: controller, 200: original}
        sent = []

        class FakeController:
            pid = 100
            done = False

            def poll(self):
                return 0 if self.done else None

            def wait(self, timeout=None):
                self.done = True
                current.pop(100, None)
                return 0

        def members(pgid):
            if pgid == 200:
                current[200] = replacement
            return [copy.deepcopy(v) for v in current.values() if v['pgid'] == pgid]

        read_text = pathlib.Path.read_text

        def text(path, *args, **kwargs):
            if str(path) == '/proc/self/cgroup':
                return controller['cgroup']
            return read_text(path, *args, **kwargs)

        plan = {'controller_argv': ['/fake/controller'], 'controller_limit_seconds': 1,
                'roles': [{'name': 'reference', 'argv': original['argv'],
                           'sha256': 'hash', 'limit_seconds': .4}]}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(watch.subprocess, 'Popen', return_value=FakeController()), \
                patch.object(watch, 'image_sha', return_value='hash'), \
                patch.object(watch, 'process_identity', side_effect=lambda pid, with_hash=False:
                             copy.deepcopy(current.get(pid))), \
                patch.object(watch, 'group_members', side_effect=members), \
                patch.object(watch, 'thread_children', return_value={101: [200]}), \
                patch.object(pathlib.Path, 'resolve', lambda path, **kwargs: path), \
                patch.object(pathlib.Path, 'read_text', text), \
                patch.object(watch.os, 'killpg', side_effect=lambda *args: sent.append(args)), \
                patch.object(watch.signal, 'pidfd_send_signal',
                             side_effect=lambda *args: sent.append(args)):
            output = pathlib.Path(directory) / 'watch.json'
            result = watch.run(plan, output)
            self.assertEqual(result, 125)
            self.assertEqual(sent, [])
            self.assertEqual(current[200]['start_ticks'], 9999)
            self.assertIn('leader changed', output.read_text())
            evidence = os.environ.get('SUPGPU_WATCHER_TEST_EVIDENCE')
            if evidence:
                path = pathlib.Path(evidence)
                path.mkdir(parents=True, exist_ok=True)
                (path / 'reused-pgid-fixed-run.json').write_bytes(output.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
