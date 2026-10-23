"""CPU placement for live UHD capture and processing."""
import os
from pathlib import Path
import shlex
import shutil


def _cpu_list(value):
    cpus = set()
    for part in value.strip().split(','):
        ends = part.split('-')
        first = int(ends[0])
        last = int(ends[-1])
        if first < 0 or last < first or len(ends) > 2:
            raise ValueError('Invalid CPU list')
        cpus.update(range(first, last + 1))
    return cpus


def capture_partition(allowed=None, topology_root=Path('/sys/devices/system/cpu')):
    """Reserve one physical core within the current CPU assignment."""
    try:
        allowed = set(os.sched_getaffinity(0) if allowed is None else allowed)
        groups = set()
        for cpu in allowed:
            path = Path(topology_root) / f'cpu{cpu}/topology/thread_siblings_list'
            siblings = _cpu_list(path.read_text())
            if cpu not in siblings:
                return None
            groups.add(frozenset(siblings & allowed))
        groups = sorted(groups, key=min)
        # Small hosts need their cores for processing. Do not guess topology.
        if len(groups) < 4:
            return None
        capture = set(groups[1])
        return capture, allowed - capture
    except (AttributeError, OSError, ValueError):
        return None


def isolate_capture_commands(capture, processing, driver):
    """Apply CPU placement before the child interpreters create threads."""
    if driver != 'uhd' or os.environ.get('LORA_CAPTURE_ISOLATION', '1') == '0':
        return capture, processing
    taskset = shutil.which('taskset')
    partition = capture_partition() if taskset else None
    if partition is None:
        return capture, processing
    capture_cpus, processing_cpus = partition

    def prefix(command, cpus):
        cpu_arg = ','.join(map(str, sorted(cpus)))
        return f'{shlex.quote(taskset)} -c {cpu_arg} {command}'

    return prefix(capture, capture_cpus), prefix(processing, processing_cpus)


def pin_reader():
    """Keep the input reader on an allowed CPU."""
    try:
        allowed = os.sched_getaffinity(0)
        if allowed:
            os.sched_setaffinity(0, {min(allowed)})
    except (AttributeError, OSError):
        pass
