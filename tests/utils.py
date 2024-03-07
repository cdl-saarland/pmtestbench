
import subprocess

def run_script(script_path, script_args, **kwargs):
    cmd = [
        "/usr/bin/env",
        "python3",
        script_path,
        *script_args
        ]
    res = subprocess.run(cmd, **kwargs)
    return res

