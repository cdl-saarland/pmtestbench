""" An interface for entities that estimate the throughput as implied by the
port mapping (i.e. without dependencies) that experiments achieve.
"""

from abc import ABC, abstractmethod
from typing import *
from datetime import datetime
from itertools import islice

from iwho.configurable import ConfigMeta

from ..experiments import Experiment
from ..architecture import Architecture
from ..portmapping import Mapping

# from itertools recipes
def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

class Processor(metaclass=ConfigMeta):
    config_options = dict(
        processor_kind = ('portmapping',
            'the kind of processor to use. One of "portmapping", "pureportmapping", "fastportmapping", "gurobiportmapping", "visportmapping", "iwho", "remote", "sharablepm", ...'), #TODO
        jitter = (None, # TODO not yet implemented
            'if not null/None but a floating point number, apply a random jitter with the given magnitude to results'),
        portmapping = (None,
            'for portmapping processor: dictionary containing a jsonified port mapping or string with a path to a portmapping file'),
        spm_path = (None,
            'for sharable portmapping processor: path to a sharable portmapping file'),
        spm_restrict = (False,
            'for sharable portmapping processor: restrict the portmapping to the instructions with non-empty port usage'),
        palmed_insnmap_path = (None,
            'for palmed processor: path to a mapping from our instruction schemes to theirs'),
        ssl_path = (None,
            'for remote processor: path to ssl key and cert files'),
        iwho_config_path = (None,
            'for iwho processor: path to the iwho config to use'),
        iwho_predictor_config = (None,
            'for iwho processor: predictor config to use'),
        iwho_unroll_factors = ([4],
            'for iwho processor: numbers of times to unroll benchmarks before doing register allocation, to reduce loop-carried dependencies. Minimum result is taken.'),
        iwho_unroll_mode = ("instructions",
            'for iwho processor: How to unroll: "simple" copies the benchmark unroll_factor times, "instructions" copies the benchmark just often enough so that >=unroll_factor instructions are in the loop body'),
        iwho_operand_allocators = (["partitioned"],
                                   'for iwho processor: list of identifiers of operand allocation strategies. For each entry, a run with operands allocated according to the strategy is performed and the minimum result is taken. Available allocators: "partitioned", "wmrr", "write-same", "random"'),
        iwho_adjust_zenp_uopcount = (False,
            'for iwho processor: HACK: adjust the uop counter for nanobench Zen+ runs for memory operands in the experiment'),
        remote_host = ("localhost", 'for remote processor: host to connect to'),
        remote_port = (17428, 'for remote processor: port to connect to'),
        remote_timeout_secs = (300, 'for remote processor: timeout in seconds for remote connection'),
        auto_split_batch = (100, 'for batch processing: automatically split iseq batches into smaller batches of at most this size'),
    )

    def __init__(self, config, *, enable_result_logging=False):
        self.enable_result_logging = enable_result_logging
        self.result_log = []

        # allow using an already constructed port mapping directly and allow
        # just using port mapping files as configs:
        if isinstance(config, Mapping) or config.get('kind', None) in ['Mapping2', 'Mapping3']:
            config = {
                    'processor_kind': 'portmapping',
                    'portmapping': config,
                }

        self.configure(config)

        if self.processor_kind.endswith('portmapping'):
            if self.portmapping is None:
                raise RuntimeError("Missing portmapping field for portmapping processor")
            elif isinstance(self.portmapping, str):
                with open(self.portmapping, 'r') as f:
                    mapping = Mapping.read_from_json(f)
            elif isinstance(self.portmapping, dict):
                pm_dict = self.portmapping
                mapping = Mapping.read_from_json_dict(pm_dict)
            else:
                mapping = self.portmapping

        if self.processor_kind == 'portmapping':
            # use the faster one if it is available
            from .portmapping_processor import has_cppfastproc
            if has_cppfastproc:
                self.processor_kind = 'nativeportmapping'
            else:
                self.processor_kind = 'pureportmapping'

        if self.processor_kind == 'pureportmapping':
            from .portmapping_processor import PurePortMappingProcessor
            self.impl = PurePortMappingProcessor(mapping)
        elif self.processor_kind == 'nativeportmapping':
            from .portmapping_processor import NativePortMappingProcessor
            self.impl = NativePortMappingProcessor(mapping)
        elif self.processor_kind == 'gurobiportmapping':
            from .gurobi_processor import GurobiProcessor
            self.impl = GurobiProcessor(mapping)
        elif self.processor_kind == 'visportmapping':
            from .vis_processor import VisProcessor
            self.impl = VisProcessor(mapping)
        elif self.processor_kind == 'sharablepm':
            from .sharablepm_processor import SharablePMProcessor
            self.impl = SharablePMProcessor(self.spm_path, self.spm_restrict)
        elif self.processor_kind == 'palmed':
            from .palmed_processor import PalmedProcessor
            self.impl = PalmedProcessor(self.get_config())
        elif self.processor_kind == 'iwho':
            from .iwho_processor import IWHOProcessor
            self.impl = IWHOProcessor(self.get_config())
        elif self.processor_kind == 'remote':
            from .remote_processor import RemoteProcessor
            self.impl = RemoteProcessor(self.get_config())
        else:
            raise NotImplementedError(f"Processor kind: {self.processor_kind}")

    def get_restricted(self, insn_list):
        """ Create a processor that only supports a subset of the instructions
        supported by this processor and behaves identically for those.
        """
        original_insns = set(self.get_arch().insn_list)
        new_insns = set(insn_list)
        if not new_insns.issubset(original_insns):
            raise ValueError("Restricted list of instructions is not a subset of the original instruction list")
        new_arch = Architecture()
        new_arch.add_insns(new_insns)
        return RestrictedProcessor(new_arch, self)


    def get_arch(self) -> Architecture:
        return self.impl.get_arch()

    def get_cycles(self, exp: Experiment) -> float:
        res = self.impl.get_cycles(exp)
        if self.enable_result_logging:
            self.result_log.append((datetime.now().isoformat(), exp, {'cycles': res}))
        return res

    def execute(self, iseq: List[str], *args, **kwargs) -> Dict[str, Union[float, str]]:
        res = self.impl.execute(iseq, *args, **kwargs)
        if self.enable_result_logging:
            self.result_log.append((datetime.now().isoformat(), iseq, res))
        return res

    def execute_batch(self, iseqs: List[List[str]], progress_callback=None, *args, **kwargs) -> List[Dict[str, Union[float, str]]]:
        # The progress_callback(num_processed, num_total, iseqs, new_results) function
        # is called regularly to report progress. It is called with the number
        # of processed benchmarks at the time, the total number of benchmarks
        # in the full batch, and the instruction sequences handled since the
        # last call and their results.

        num_processed = 0
        num_total = len(iseqs)

        if progress_callback is None:
            progress_callback = lambda num_processed, num_total, iseqs, res: None

        progress_callback(num_processed, num_total, [], [])
        full_res = []
        for batch in batched(iseqs, self.auto_split_batch):
            res = self.impl.execute_batch(batch, *args, **kwargs)

            full_res.extend(res)

            if self.enable_result_logging:
                for iseq, r in zip(batch, res):
                    self.result_log.append((datetime.now().isoformat(), iseq, r))

            num_processed += len(batch)
            progress_callback(num_processed, num_total, batch, res)

        return full_res

    def eval(self, exp: Experiment):
        """ Evaluate the given experiment and insert the results.
        """
        res = self.execute(exp.iseq)
        exp.result = res

    def eval_batch(self, exps, recompute=True, progress_callback=None):
        if not recompute:
            exps = [e for e in exps if e.result is None]

        remaining_exps = [e for e in exps]

        def wrapping_callback(num_processed, num_total, iseqs, res, *, progress_callback=progress_callback, remaining_exps=remaining_exps):
            processed_in_this_batch = len(iseqs)
            for e, r in zip(remaining_exps[:processed_in_this_batch], res):
                e.result = r
            del remaining_exps[:processed_in_this_batch]
            if progress_callback is not None:
                progress_callback(num_processed, num_total, iseqs, res)

        results = self.execute_batch([e.iseq for e in exps], progress_callback=wrapping_callback)

        # this is already done by the callback
        # for e, r in zip(exps, results):
        #     e.result = r

    # def eval_list(self, exps: ExperimentList):
    #     """ Evaluate the given ExperimentList and insert the results.
    #     """
    #     for e in exps:
    #         self.eval(e)

class RestrictedProcessor:
    """ A wrapper for a Processor object to restrict the architecture to a
    subset of instructions.
    """
    def __init__(self, restricted_arch, wrapped_processor):
        self.restricted_arch = restricted_arch
        self.wrapped_processor = wrapped_processor

    def get_arch(self) -> Architecture:
        return self.restricted_arch

    def get_cycles(self, exp: Experiment) -> float:
        return self.wrapped_processor.get_cycles(exp)

    def execute(self, iseq: List[str], *args, **kwargs) -> Dict[str, Union[float, str]]:
        return self.wrapped_processor.execute(iseq, *args, **kwargs)

    def eval(self, exp: Experiment):
        self.wrapped_processor.eval(exp)



class ProcessorImpl(ABC):

    @abstractmethod
    def get_arch(self) -> Architecture:
        pass

    def execute(self, iseq: List[str], *args, **kwargs) -> Dict[str, Union[float, str]]:
        """ Return a dictionary with execution results for the list iseq of
        instructions. The result has to contain at least a float entry for the
        key 'cycles'.

        At least this method or get_cycles need to be overridden by an
        implementation.
        """
        res = self.get_cycles(iseq)
        return { 'cycles': res }

    def execute_batch(self, iseqs: List[List[str]], *args, **kwargs) -> List[Dict[str, Union[float, str]]]:
        """ Return a list of dictionaries with execution results for the list
        iseq of instructions. The result has to contain at least a float entry
        for the key 'cycles'.
        """
        return [self.execute(iseq, *args, **kwargs) for iseq in iseqs]

    def get_cycles(self, iseq: List[str]) -> float:
        """ Return the number of cycles required to execute the list iseq of
        instructions.

        At least this method or execute need to be overridden by an
        implementation.
        """
        res = self.execute(iseq)
        return res['cycles']

