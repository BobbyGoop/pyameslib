import datetime
import functools
import itertools
import json
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Semaphore
from typing import Dict, List, Callable

logging.basicConfig(level=logging.INFO)

import sh


def custom_log(ran, call_args, pid=None):
    return ran


here = os.path.dirname(__file__)

cores = os.cpu_count()

if not cores:
    cores = 16

_threads_singleprocess = cores  # 24, 16
_threads_multiprocess = (
    _threads_singleprocess // 2 if _threads_singleprocess > 1 else 1
)  # 12, 8
_processes = _threads_multiprocess // 4 if _threads_multiprocess > 3 else 1  # 3, 2

pool = Semaphore(cores)

__reserved_kwargs_for_asap = ["postfix"]


def done(cmd, success, exit_code):
    pool.release()


def circ_mean(*vargs, low=-180.0, high=180.0):
    lowr, highr = math.radians(low), math.radians(high)
    # based on scipy's circ_mean
    vargs_rads = list(map(math.radians, vargs))
    vargs_rads_subr = [(_ - lowr) * 2.0 * math.pi / (highr - lowr) for _ in vargs_rads]
    sinsum = sum(list(map(math.sin, vargs_rads_subr)))
    cossum = sum(list(map(math.cos, vargs_rads_subr)))
    res = math.atan2(sinsum, cossum)
    if res < 0:
        res += 2 * math.pi
    res = math.degrees(res * (highr - lowr) / 2.0 / math.pi + lowr)
    return res


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    try:
        if newdir:
            os.chdir(newdir)
            print(f"cd {newdir}", flush=True)
        yield
    finally:
        os.chdir(prevdir)
        print(f"cd {prevdir}", flush=True)


@contextmanager
def silent_cd(newdir):
    prevdir = os.getcwd()
    try:
        if newdir:
            os.chdir(newdir)
        yield
    finally:
        os.chdir(prevdir)


def optional(variable, null=""):
    # TODO: this is equivalent to something from functional programming that I am forgetting the name of
    if isinstance(variable, (bool, int, float, str, Path)):
        variable = [variable]
    for _ in variable:
        if _ != null:
            yield _


def cmd_to_string(command: sh.RunningCommand) -> str:
    """
    Converts the running command into a single string of the full command call for easier logging

    :param command: a command from sh.py that was run
    :return: string of bash command
    """
    return " ".join((_.decode("utf-8") for _ in command.cmd))


def clean_args(args):
    return list(
        itertools.chain.from_iterable(
            [x.split(" ") if isinstance(x, str) else (x,) for x in args]
        )
    )


def clean_kwargs(kwargs: Dict) -> Dict:
    # remove any reserved asap kwargs
    for rkw in __reserved_kwargs_for_asap:
        kwargs.pop(rkw, None)
    cleaned = {}
    for key in kwargs.keys():
        new_key = str(key)
        if not key.startswith("--") and len(key) > 1:
            new_key = f"--{key}"
        elif not key.startswith("-"):
            new_key = f"-{key}"
        new_key = new_key.replace("_", "-")
        cleaned[new_key] = kwargs[key]
    return cleaned


def convert_kwargs(kwargs: Dict) -> List:
    keys = []
    # ensure keys start with '--' for asp scripts
    for key in kwargs.keys():
        if not isinstance(key, str):
            raise ValueError('Shell command must be string, starting with `-` or `--`')
        if key not in ("--t_srs", "--t_projwin"):
            key = key.replace("_", "-")
        if not key.startswith("--") and len(key) > 1:
            keys.append(f"--{key}")
        elif not key.startswith("-"):
            keys.append(f"-{key}")
        else:
            keys.append(key)

    # Also convert key values to string
    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(keys, map(str, kwargs.values()))
        )
        if x is not None
    ]


def parse_kwargs(kwargs: Dict, key: str) -> str:
    if kwargs is None:
        return ""
    key_args = kwargs.get(key, {})
    if isinstance(key_args, str):
        return key_args
    return " ".join(map(str, convert_kwargs(clean_kwargs(key_args))))


def get_affine_from_file(file):
    import affine

    md = json.loads(str(sh.gdalinfo(file, "-json")))
    gt = md["geoTransform"]
    return affine.Affine.from_gdal(*gt)


def rich_logger(func: Callable):
    """
    rich logger decorator, wraps a function and writes nice log statements

    :param func: function to wrap
    :return: wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        # grab the function name first
        func_name = func.__name__
        # check if we are running a sh/bash command or a normal python function
        if "/bin/" not in func_name:
            if func.__doc__ is None:
                name_line = f"{func_name}"
            else:
                # grab the first doc line for the pretty name, make sure all functions have docs!
                pretty_name = func.__doc__.splitlines()[1].strip()
                # generate the name line
                name_line = f"{func_name} ({pretty_name})"
        else:
            # else we have a bash command and won't have a pretty name
            name_line = func_name
        # log out the start line with the name line and start time
        print(
            f"""# Started: {name_line}, at: {start_time.isoformat(" ")}""", flush=True
        )
        # call the function and get the return
        ret = func(*args, **kwargs)
        # if we had a running command log out the call
        if ret is not None and isinstance(ret, sh.RunningCommand):
            print(f"# Ran Command: {cmd_to_string(ret)}", flush=True)
        # else just get the time duraton
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        # log out the execution time
        print(
            f"""# Finished: {name_line}, at: {end_time.isoformat(" ")}, duration: {str(duration)}""",
            flush=True,
        )
        # no return at this point

    return wrapper


def run_parallel(func, all_calls_args):
    """
    Parallel execution helper function for sh.py Commands

    :param func: func to call
    :param all_calls_args: args to pass to each call of the func
    :return: list of called commands
    """
    procs = []

    def do(*args):
        pool.acquire()
        return func(*args, _bg=True, _done=done)

    for call_args in all_calls_args:
        if " " in call_args:
            # if we are running a command with multiple args, sh needs different strings
            call_args = call_args.split(" ")
            procs.append(do(*call_args))
        else:
            procs.append(do(call_args))

    return [p.wait() for p in procs]
