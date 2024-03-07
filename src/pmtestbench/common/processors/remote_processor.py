""" A processor using rpyc to access a remote benchmarking server."""

import math
import numbers
from typing import *
import os
import random
import sys

import rpyc

from . import ProcessorImpl

from ..architecture import Architecture

from iwho.configurable import load_json_config
import iwho


import logging
logger = logging.getLogger(__name__)

def unwrap(x):
    if isinstance(x, list):
        return [ unwrap(y) for y in x ]
    elif isinstance(x, dict):
        return { k: unwrap(x[k]) for k in x }
    else:
        return x

class RemoteProcessor(ProcessorImpl):

    def __init__(self, config):
        self.host = config['remote_host']
        self.port = config['remote_port']
        self.timeout_secs = config['remote_timeout_secs']

        config_path = config['iwho_config_path']
        if config_path is not None:
            if isinstance(config_path, dict):
                # embedded config
                iwhoconfig = config_path
            else:
                iwhoconfig = load_json_config(config_path)
        else:
            iwhoconfig = {} # use the defaults
        self.iwho_ctx = iwho.Config(config=iwhoconfig).context

        insn_schemes = list(map(str, self.iwho_ctx.filtered_insn_schemes))

        insn_set = set(insn_schemes)

        with rpyc.connect(self.host, self.port) as conn:
            wrapped_res = conn.root.get_insns()
            res = unwrap(wrapped_res)

        remote_insns = set(res)

        if not insn_set.issubset(remote_insns):
            raise ValueError("The remote server does not support the following instructions: {}".format(
                insn_set - remote_insns))

        self.arch = Architecture()
        self.arch.add_insns(insn_schemes)



    def get_arch(self) -> Architecture:
        return self.arch


    def execute(self, iseq: List[str], *args, excessive_log=None, **kwargs) -> Dict[str, Union[float, str]]:
        return self.execute_batch([iseq], *args, excessive_log=excessive_log, **kwargs)[0]

    def execute_batch(self, iseqs: List[List[str]], *args, **kwargs) -> List[Dict[str, Union[float, str]]]:
        with rpyc.connect(self.host, self.port, config={"sync_request_timeout": self.timeout_secs}) as conn:
            wrapped_res = conn.root.execute_batch(iseqs, *args, **kwargs)
            res = unwrap(wrapped_res)
        return res


