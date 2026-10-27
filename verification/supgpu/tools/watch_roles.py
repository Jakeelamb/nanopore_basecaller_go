#!/usr/bin/python3
"""Bound the exact native-role children of a Go controller; never select work."""

import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_stat(pid):
    try:
        raw = pathlib.Path('/proc/%d/stat' % pid).read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rindex(')') + 1:].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'pgid': int(fields[2]), 'sid': int(fields[3]),
            'start_ticks': int(fields[19]), 'raw': raw}


def thread_children(pid):
    """Linux children belongs to the creating task, not its thread-group leader."""
    by_thread = {}
    for path in pathlib.Path('/proc/%d/task' % pid).glob('*/children'):
        try:
            by_thread[int(path.parent.name)] = [int(v) for v in path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            continue
    return by_thread


def group_members(pgid):
    found = []
    for path in pathlib.Path('/proc').glob('[0-9]*/stat'):
        record = process_stat(int(path.parent.name))
        if record and record['pgid'] == pgid:
            found.append(record)
    return found


def image_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


class OwnershipError(RuntimeError):
    pass


def process_identity(pid, with_hash=False):
    record = process_stat(pid)
    if not record or record['state'] == 'Z':
        return None
    try:
        record['image'] = str(pathlib.Path('/proc/%d/exe' % pid).resolve(strict=True))
        raw = pathlib.Path('/proc/%d/cmdline' % pid).read_bytes()
        record['argv'] = [v.decode() for v in raw.split(b'\0')[:-1]]
        record['cgroup'] = pathlib.Path('/proc/%d/cgroup' % pid).read_text()
        if with_hash:
            record['image_sha256'] = image_sha('/proc/%d/exe' % pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    after = process_stat(pid)
    if not after or any(record[k] != after[k] for k in ('start_ticks', 'pgid', 'sid')):
        raise OwnershipError('process changed while reading identity: %d' % pid)
    return record


def same_identity(expected, actual, with_hash=False):
    keys = ['pid', 'start_ticks', 'pgid', 'sid', 'image', 'argv', 'cgroup']
    if with_hash:
        keys.append('image_sha256')
    return actual is not None and all(expected[k] == actual.get(k) for k in keys)


class OwnedGroup:
    """Track a private group; never signal a cached numeric PGID.

    A live authenticated leader can establish descendant ancestry. After it
    exits, only already authenticated members retain ownership. Each signal
    uses a pidfd opened before rechecking the member's complete identity.
    """
    def __init__(self, leader, signals):
        if leader['pgid'] != leader['pid']:
            raise OwnershipError('leader does not own a private process group')
        self.leader = leader
        self.pgid = leader['pid']
        self.members = {leader['pid']: leader}
        self.signals = signals
        self.retired = False

    def refresh(self):
        if self.retired:
            return []
        records = group_members(self.pgid)
        current = {}
        for record in records:
            if record['state'] == 'Z':
                continue
            actual = process_identity(record['pid'])
            if actual is None:
                continue
            if (actual['start_ticks'], actual['pgid']) != (record['start_ticks'], self.pgid):
                raise OwnershipError('group member changed during observation')
            current[actual['pid']] = actual
        leader = current.get(self.pgid)
        if leader and not same_identity(self.leader, leader):
            self.retired = True
            raise OwnershipError('private group leader changed; numeric group retired')
        for pid, actual in current.items():
            if pid in self.members and not same_identity(self.members[pid], actual):
                raise OwnershipError('authenticated member identity changed: %d' % pid)
        pending = {pid for pid in current if pid not in self.members}
        if pending and not leader:
            raise OwnershipError('unproven member after leader exit; outer cgroup cleanup required')
        while pending:
            proved = []
            for pid in pending:
                actual = current[pid]
                parent = actual['ppid']
                if parent not in current or parent not in self.members:
                    continue
                if (actual['sid'], actual['cgroup']) != (self.leader['sid'], self.leader['cgroup']):
                    raise OwnershipError('descendant session/cgroup changed')
                bound = process_identity(pid, with_hash=True)
                if not same_identity(actual, bound):
                    raise OwnershipError('descendant identity changed during authentication')
                self.members[pid] = bound
                proved.append(pid)
            if not proved:
                raise OwnershipError('unproven member ancestry; outer cgroup cleanup required')
            pending.difference_update(proved)
        if not current:
            self.retired = True
        return list(current.values())

    def signal(self, sig):
        # Validate the whole observed membership before signaling any member.
        active = self.refresh()
        for member in active:
            pid = member['pid']
            try:
                fd = os.pidfd_open(pid, 0)
            except ProcessLookupError:
                continue
            except OSError as error:
                raise OwnershipError('pidfd_open failed; no numeric signal fallback') from error
            try:
                actual = process_identity(pid, with_hash=True)
                if actual is None:
                    continue
                if not same_identity(self.members[pid], actual, with_hash=True):
                    raise OwnershipError('member identity changed after pidfd_open: %d' % pid)
                event = {'pid': pid, 'pgid': self.pgid, 'signal': sig, 'at': utc(),
                         'pidfd': True, 'identity': actual}
                self.signals.append(event)
                try:
                    signal.pidfd_send_signal(fd, sig)
                    event['sent'] = True
                except ProcessLookupError:
                    event['sent'] = False
                except OSError as error:
                    raise OwnershipError('pidfd signal failed; outer cgroup cleanup required') from error
            finally:
                os.close(fd)


def run(plan, output):
    roles = plan['roles']
    if not roles or len(roles) > 8 or len({r['name'] for r in roles}) != len(roles):
        raise ValueError('one to eight unique expected roles required')
    for role in roles:
        if not role['argv'] or not 0 < role['limit_seconds'] <= 180:
            raise ValueError('exact role argv and bounded lifetime required')
        if str(pathlib.Path(role['argv'][0]).resolve()) != role['argv'][0]:
            raise ValueError('role image must be canonical')
        if image_sha(role['argv'][0]) != role['sha256']:
            raise ValueError('prospective role image changed before controller')
    if not 0 < plan['controller_limit_seconds'] <= 560:
        raise ValueError('controller limit must fit outer600s cleanup budget')
    controller_image = str(pathlib.Path(plan['controller_argv'][0]).resolve(strict=True))
    controller_sha = image_sha(controller_image)
    own_cgroup = pathlib.Path('/proc/self/cgroup').read_text()
    receipt = {'schema': 'nanopore-native-role-watch/v1', 'started': utc(),
               'plan': plan, 'roles': [], 'signals': [], 'timeout': False,
               'ownership_errors': [], 'own_cgroup': own_cgroup,
               'signal_policy': 'current-identity-revalidated-pidfd-members-only'}
    proc = subprocess.Popen(plan['controller_argv'], start_new_session=True)
    receipt['controller_pid'] = receipt['controller_pgid'] = proc.pid
    began = time.monotonic()
    hz = os.sysconf('SC_CLK_TCK')
    seen = {}
    groups = {}
    failure = None

    def stop(group, sig):
        try:
            group.signal(sig)
        except OwnershipError as error:
            receipt['ownership_errors'].append({'pgid': group.pgid, 'error': str(error), 'at': utc()})

    def wait_controller(seconds):
        try:
            proc.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

    try:
        controller = process_identity(proc.pid, with_hash=True)
        if (not controller or controller['pgid'] != proc.pid or controller['sid'] != proc.pid
                or controller['image'] != controller_image
                or controller['argv'] != plan['controller_argv']
                or controller['image_sha256'] != controller_sha
                or controller['cgroup'] != own_cgroup):
            raise OwnershipError('controller birth/image/argv/private group/session/cgroup not authenticated')
        receipt['controller_identity'] = controller
        groups[proc.pid] = OwnedGroup(controller, receipt['signals'])
        while proc.poll() is None:
            by_thread = thread_children(proc.pid)
            children = sorted({child for group in by_thread.values() for child in group})
            for pid in children:
                if pid in seen:
                    continue
                record = process_identity(pid, with_hash=True)
                if not record:
                    continue
                argv = record['argv']
                matches = [r for r in roles if r['argv'] == argv and r['argv'][0] == record['image']]
                if (len(matches) != 1 or record['ppid'] != proc.pid or record['pgid'] != pid
                        or record['sid'] != controller['sid'] or record['cgroup'] != own_cgroup):
                    raise RuntimeError('unexpected child identity/argv/private group: %r' % record)
                role = matches[0]
                if role['name'] in [v['name'] for v in seen.values()]:
                    raise RuntimeError('duplicate role execution')
                if role is not roles[len(seen)]:
                    raise RuntimeError('native roles arrived out of the declared order')
                if record['image_sha256'] != role['sha256']:
                    raise RuntimeError('actual child image hash differs')
                entry = {'name': role['name'], 'pid': pid, 'pgid': pid, 'argv': argv,
                         'image_sha256': role['sha256'], 'observed_started': utc(),
                         'start_ticks': record['start_ticks'], 'clock_ticks_per_second': hz,
                         'sid': record['sid'], 'cgroup': record['cgroup'],
                         'stat': record['raw'], 'limit_seconds': role['limit_seconds'],
                         'children_by_thread': by_thread,
                         'old_main_thread_children': by_thread.get(proc.pid, [])}
                seen[pid] = entry
                groups[pid] = OwnedGroup(record, receipt['signals'])
                receipt['roles'].append(entry)
            for pid, entry in seen.items():
                active = groups[pid].refresh()
                if not active:
                    entry.setdefault('observed_finished', utc())
                    continue
                age = time.clock_gettime(time.CLOCK_BOOTTIME) - entry['start_ticks'] / hz
                if age >= entry['limit_seconds']:
                    entry['timeout_age_seconds'] = age
                    entry['members_at_timeout'] = active
                    raise TimeoutError('role %s exceeded its birth-relative limit' % entry['name'])
            if time.monotonic() - began >= plan['controller_limit_seconds']:
                raise TimeoutError('controller exceeded its bound')
            time.sleep(.02)
    except Exception as error:
        failure = str(error)
        receipt['error'] = failure
        receipt['timeout'] = isinstance(error, TimeoutError)
        if isinstance(error, OwnershipError):
            receipt['ownership_errors'].append({'error': failure, 'at': utc()})
        for pid in seen:
            stop(groups[pid], signal.SIGTERM)
        if not wait_controller(5):
            if proc.pid in groups:
                stop(groups[proc.pid], signal.SIGTERM)
            wait_controller(5)
            for pid in seen:
                stop(groups[pid], signal.SIGKILL)
            if proc.pid in groups:
                stop(groups[proc.pid], signal.SIGKILL)
            wait_controller(5)
    receipt['controller_exit'] = proc.poll()
    # Group cleanup is independent of leader Wait and pipe inheritance.
    receipt['members_after_leader'] = {str(pid): group_members(pid) for pid in [proc.pid, *seen]}
    for pid, group in groups.items():
        try:
            active = group.refresh()
        except OwnershipError as error:
            receipt['ownership_errors'].append({'pgid': pid, 'error': str(error), 'at': utc()})
            continue
        if active:
            stop(group, signal.SIGKILL)
            for _ in range(100):
                try:
                    if not group.refresh():
                        break
                except OwnershipError as error:
                    receipt['ownership_errors'].append({'pgid': pid, 'error': str(error), 'at': utc()})
                    break
                time.sleep(.05)
    receipt['members_after_cleanup'] = {str(pid): group_members(pid) for pid in [proc.pid, *seen]}
    if any(receipt['members_after_cleanup'].values()) or receipt['controller_exit'] is None:
        failure = 'processes remain; outer owned-cgroup cleanup required'
        receipt['cleanup_error'] = failure
    if receipt['ownership_errors']:
        failure = failure or 'process ownership became ambiguous; no cached group signal allowed'
    if not failure and receipt['controller_exit'] == 0 and len(seen) != len(roles):
        failure = 'successful controller did not expose every expected role'
        receipt['error'] = failure
    receipt['finished'] = utc()
    with pathlib.Path(output).open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    return 124 if receipt['timeout'] else 125 if failure else receipt['controller_exit']


if __name__ == '__main__':
    if len(sys.argv) != 4 or image_sha(sys.argv[1]) != sys.argv[3]:
        raise SystemExit('independent expected role-plan SHA256 required')
    with open(sys.argv[1]) as plan_file:
        plan = json.load(plan_file)
    raise SystemExit(run(plan, sys.argv[2]))
