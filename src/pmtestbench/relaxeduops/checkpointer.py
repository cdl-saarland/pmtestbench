""" A helper utility to split the core algorithm into stages and store
intermediate results to restart based on them.
"""

from copy import deepcopy
import math
import pickle
import time

import logging
logger = logging.getLogger(__name__)

class CheckPointer:
    """ This class provides a method to store and load intermediate results
    from a number of stages.
    """

    class CheckPointSkip:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    class CheckPoint:
        def __init__(self, checkpointer, prev, preserve_all):
            self._has_run = False
            self._has_stored = False
            self._checkpointer = checkpointer
            self._res_data = dict()
            self._prev = prev
            self._start_time = None
            self._end_time = None
            if preserve_all and self._prev is not None:
                for key, value in self._prev._res_data.items():
                    setattr(self, key, value)
            self._logger = self._checkpointer.logger

        def has_key(self, key):
            return key in self._res_data.keys()

        def __iter__(self):
            return self

        def __setattr__(self, key, new_value):
            super().__setattr__(key, new_value)
            if not key.startswith('_'):
                self._res_data[key] = new_value

        def __next__(self):
            if self._has_run:
                if self._has_stored:
                    raise RuntimeError("checkpoint run after it was already terminated")
                # execution of the stage is considered to end now
                self.end_time = time.perf_counter()
                timediff_secs = self.end_time - self.start_time
                self._logger.info(f"STAGE {self._checkpointer.curr_stage} performed in {timediff_secs} seconds")

                # store the data
                self._checkpointer.store(self._res_data)

                self._has_stored = True
                raise StopIteration

            self._has_run = True
            self.start_time = time.perf_counter()
            # execution of the stage is considered to start now
            return self._prev, self


    def __init__(self, *, start_with=None, run_steps=None, checkpoint_base=None, store_dir=None, preserve_all=True, custom_logger=None):
        self.start_with = start_with
        self.store_dir = store_dir
        self.preserve_all = preserve_all
        self.logger = custom_logger or logger

        self.run_steps = run_steps
        if self.run_steps is None:
            self.run_steps = math.inf

        self.checkpoint_base = checkpoint_base
        if self.checkpoint_base is None:
            self.checkpoint_base = dict()

        self.has_hit_start_stage = self.start_with is None

        self.steps_done_so_far = 0
        self.curr_stage_idx = 0
        self.curr_stage = None
        self.prev = None

    def store(self, data):
        if self.store_dir is None:
            return

        with open(self.store_dir / f'checkpoint_{self.curr_stage_idx:02}_{self.curr_stage}.pickle', 'wb') as f:
            checkpoint = deepcopy(self.checkpoint_base)
            checkpoint['stage'] = self.curr_stage
            checkpoint['stage_idx'] = self.curr_stage_idx
            checkpoint['stored_data'] = data
            pickle.dump(checkpoint, f)

    def __call__(self, stage_name):
        self.curr_stage_idx += 1
        self.curr_stage = stage_name

        if self.steps_done_so_far >= self.run_steps:
            self.logger.info(f"skipping STAGE {self.curr_stage}")
            return CheckPointer.CheckPointSkip()

        if self.has_hit_start_stage:
            self.logger.info(f"start STAGE {self.curr_stage}")

            new_cp = CheckPointer.CheckPoint(self, self.prev, preserve_all=self.preserve_all)

            self.prev = new_cp # this sets up `prev` for the stage after the current one
            self.steps_done_so_far += 1

            # let them actually do the thing
            return new_cp

        elif self.curr_stage == self.start_with['stage']:
            self.logger.info(f"resuming after STAGE {self.curr_stage}")
            # load from the checkpoint
            new_cp = CheckPointer.CheckPoint(self, None, preserve_all=self.preserve_all)
            for k, v in self.start_with['stored_data'].items():
                setattr(new_cp, k, v)
            self.prev = new_cp
            self.has_hit_start_stage = True
            return CheckPointer.CheckPointSkip()

        else:
            self.logger.info(f"skipping STAGE {self.curr_stage}")
            return CheckPointer.CheckPointSkip()


