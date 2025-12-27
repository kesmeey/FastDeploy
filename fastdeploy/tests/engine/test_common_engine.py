"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import time
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.utils import envs

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestCommonEngine(unittest.TestCase):
    """Test case for EngineService functionality (lines 1215-1664)"""

    @classmethod
    def setUpClass(cls):
        """Set up EngineService for testing"""
        try:
            # Create engine args for testing
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 10,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 10,
            )

            # Create and start the engine service
            cls.cfg = engine_args.create_engine_config()
            cls.engine = EngineService(cls.cfg, start_queue=True, use_async_llm=True)

            # Start the engine service
            cls.engine.start()

        except Exception as e:
            print(f"Setting up EngineService failed: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if hasattr(cls, "engine") and cls.engine is not None:
            try:
                cls.engine._exit_sub_services()
                print("Engine cleanup completed")
            except Exception as e:
                print(f"Error during engine cleanup: {e}")

    def setUp(self):
        """Set up before each test method"""
        print(f"Starting test: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test method"""
        print(f"Completed test: {self._testMethodName}")

    def test_exit_sub_services(self):
        """Test _exit_sub_services method (lines 1215-1291)"""
        # Test that _exit_sub_services can be called without error
        # Note: We won't actually call it since it would shut down the engine
        # Instead we'll test that the method exists and has expected attributes
        self.assertTrue(hasattr(self.engine, "_exit_sub_services"))
        self.assertTrue(callable(getattr(self.engine, "_exit_sub_services")))

        # Test that engine has expected attributes that would be cleaned up
        if hasattr(self.engine, "worker_proc"):
            self.assertIsNotNone(self.engine.worker_proc)

        # Verify running state
        self.assertTrue(self.engine.running)

    def test_worker_processes_ready(self):
        """Test _worker_processes_ready method (lines 1292-1299)"""
        # Test with real engine that should have worker_ready_signal
        if hasattr(self.engine, "worker_ready_signal"):
            result = self.engine._worker_processes_ready()
            # Result should be boolean
            self.assertIsInstance(result, bool)
        else:
            self.skipTest("worker_ready_signal not available")

    def test_init_worker_signals(self):
        """Test _init_worker_signals method (lines 1301-1361)"""
        # Since engine is already started, signals should be initialized
        self.assertTrue(hasattr(self.engine, "worker_ready_signal"))
        self.assertTrue(hasattr(self.engine, "loaded_model_signal"))

        # Test that signals have expected properties
        if hasattr(self.engine, "worker_ready_signal"):
            self.assertIsNotNone(self.engine.worker_ready_signal)

        if hasattr(self.engine, "loaded_model_signal"):
            self.assertIsNotNone(self.engine.loaded_model_signal)

    def test_setting_environ_variables(self):
        """Test _setting_environ_variables method (lines 1362-1408)"""
        result = self.engine._setting_environ_variables()

        # Check that result is a string and contains expected variables
        self.assertIsInstance(result, str)
        self.assertIn("ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY=0", result)
        self.assertIn("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python", result)
        self.assertIn("FLAGS_use_append_attn=1", result)
        self.assertIn("NCCL_ALGO=Ring", result)

    def test_start_worker_service(self):
        """Test _start_worker_service method (lines 1409-1517)"""
        # Since engine is already started, we can test that worker process exists
        if hasattr(self.engine, "worker_proc") and self.engine.worker_proc:
            # Worker process should be running
            self.assertIsNotNone(self.engine.worker_proc)
            # Process should be alive (poll returns None if still running)
            poll_result = self.engine.worker_proc.poll()
            if poll_result is not None:
                self.skipTest("Worker process is not running")
        else:
            self.skipTest("Worker process not available")

    def test_stop_profile(self):
        """Test _stop_profile method (lines 1519-1532)"""
        # Test method exists and is callable
        self.assertTrue(hasattr(self.engine, "_stop_profile"))
        self.assertTrue(callable(getattr(self.engine, "_stop_profile")))

        # We won't actually call it as it modifies engine state
        # Just verify the do_profile attribute exists
        self.assertTrue(hasattr(self.engine, "do_profile"))

    def test_check_health(self):
        """Test check_health method (lines 1533-1544)"""
        if hasattr(self.engine, "worker_healthy_live_signal"):
            is_healthy, message = self.engine.check_health(time_interval_threashold=30)

            # Should return tuple of (bool, str)
            self.assertIsInstance(is_healthy, bool)
            self.assertIsInstance(message, str)
        else:
            self.skipTest("worker_healthy_live_signal not available")

    def test_launch_components(self):
        """Test launch_components method (lines 1545-1605)"""
        # Method should exist and be callable
        self.assertTrue(hasattr(self.engine, "launch_components"))
        self.assertTrue(callable(getattr(self.engine, "launch_components")))

        # Test that scheduler exists (should be created during start)
        if hasattr(self.engine, "scheduler"):
            self.assertIsNotNone(self.engine.scheduler)

    def test_check_worker_initialize_status(self):
        """Test check_worker_initialize_status method (lines 1606-1663)"""
        # Method should exist and be callable
        self.assertTrue(hasattr(self.engine, "check_worker_initialize_status"))
        self.assertTrue(callable(getattr(self.engine, "check_worker_initialize_status")))

        # Test that worker_init_status exists
        if hasattr(self.engine, "worker_init_status"):
            self.assertIsInstance(self.engine.worker_init_status, dict)

    def test_engine_started_successfully(self):
        """Test that engine started successfully and has expected state"""
        # Verify engine is running
        self.assertTrue(self.engine.running)

        # Verify data processor was created
        if hasattr(self.engine, "data_processor"):
            self.assertIsNotNone(self.engine.data_processor)

        # Verify IPC signal suffix is set
        if hasattr(self.engine, "ipc_signal_suffix"):
            self.assertIsNotNone(self.engine.ipc_signal_suffix)


if __name__ == "__main__":
    unittest.main()


class TestCommonEngineAdditionalCoverage(unittest.TestCase):
    """Additional unit tests focusing on branch coverage for common_engine.py

    These tests heavily mock subprocess/threading/IPC to avoid starting real workers
    and to drive specific code paths that were previously uncovered.
    """

    def setUp(self):
        patch("fastdeploy.engine.common_engine.EngineCacheQueue").start()
        self.addCleanup(patch.stopall)

    def _make_cfg(self, **kwargs):
        # If DP > 1, we must provide enough engine_worker_queue_port for each dp index
        dp = kwargs.get("data_parallel_size", 1)
        nnode = len(kwargs.get("ips", ["127.0.0.1"]))
        engine_worker_queue_port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
        cache_queue_port = int(os.getenv("FD_CACHE_QUEUE_PORT", "6779"))
        if dp and dp > 1:
            engine_worker_queue_port = [engine_worker_queue_port + 20 + i for i in range(dp // nnode)]
            cache_queue_port = [cache_queue_port + 20 + i for i in range(dp // nnode)]

        args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=128,
            tensor_parallel_size=1,
            # give unique ports to avoid collision with other tests
            engine_worker_queue_port=engine_worker_queue_port,
            cache_queue_port=cache_queue_port,
            enable_prefix_caching=True,
            **kwargs,
        )
        # Keep batch tokens small to satisfy FDConfig checks:
        # max_num_batched_tokens <= max_model_len * max_num_seqs
        if getattr(args, "max_num_batched_tokens", None) is None:
            args.max_num_batched_tokens = 128
        # Always enable chunked prefill in tests to avoid another strict check
        args.enable_chunked_prefill = True

        return args.create_engine_config()

    def _make_request(self, rid: str, prompt_len: int = 4):
        return Request(
            request_id=rid,
            prompt=None,
            prompt_token_ids=None,
            prompt_token_ids_len=prompt_len,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=None,
        )

    def _stub_processor(self):
        class _Tok:
            def __init__(self):
                self.vocab = {"</think>": 42, "\n": 10, "<|IMAGE_PLACEHOLDER|>": 9}

            def get_vocab(self):
                return self.vocab

        class _Proc:
            def __init__(self):
                self.tokenizer = _Tok()
                self.eos_token_id_len = 1
                self.pad_token_id = 0

        return _Proc()

    # ---- Minimal EngineService stubs for light‑weight logic tests ----
    def _make_minimal_engine(self):
        """Create a minimal EngineService-like object built via __new__.

        We only attach attributes that the tested methods will touch,
        and use simple mocks to avoid starting any real services.
        """

        class _Cfg:
            class _SchedulerCfg:
                def __init__(self):
                    self.splitwise_role = "prefill"
                    self.max_num_batched_tokens = 16

            class _CacheCfg:
                def __init__(self):
                    self.block_size = 4
                    self.enc_dec_block_num = 0
                    self.enable_prefix_caching = True
                    self.enable_chunked_prefill = True
                    self.max_block_num_per_seq = 4

            class _ParallelCfg:
                def __init__(self):
                    self.tensor_parallel_size = 1
                    self.data_parallel_size = 1
                    self.local_data_parallel_id = 0

            class _ModelCfg:
                def __init__(self):
                    self.enable_mm = False

            class _SpeculativeCfg:
                def __init__(self):
                    self.method = None
                    self.to_json_string = Mock(return_value='{"test": true}')

            def __init__(self):
                self.scheduler_config = self._SchedulerCfg()
                self.cache_config = self._CacheCfg()
                self.parallel_config = self._ParallelCfg()
                self.model_config = self._ModelCfg()
                self.speculative_config = self._SpeculativeCfg()

        eng = EngineService.__new__(EngineService)
        eng.cfg = _Cfg()
        eng.llm_logger = Mock()
        eng.engine_worker_queue = Mock()
        eng.resource_manager = Mock()
        eng.scheduler = Mock()
        eng.split_connector = Mock()
        eng.token_processor = Mock()
        eng.token_processor.number_of_tasks = 0
        eng.token_processor.number_of_input_tokens = 0

        # Common defaults used by several methods
        eng.resource_manager.stop_flags = np.array([1, 1], dtype=np.int32)
        eng.resource_manager.real_bsz = 1
        # Provide defaults expected by reverted EngineService implementation
        eng.partial_chunked_tokens = [0, 4, 4, 4]
        eng.token_processor.tokens_counter = {}
        # Ensure split_connector.check_decode_allocated returns a tuple by default
        eng.split_connector.check_decode_allocated = lambda task: (True, "")
        return eng

    def test_start_prefill_branch_cache_manager_and_worker_dead(self):
        """Cover lines 184-185, 194-197, 221, 226-227 in start()."""
        # For prefill + local scheduler the core code now requires a router.
        # Also, with the newer CacheConfig semantics we must ensure that
        # prefill_kvcache_block_num (num_gpu_blocks_override * kv_cache_ratio)
        # is >= max_block_num_per_seq; use 3 blocks so that with the default
        # kv_cache_ratio=0.75 we still satisfy the assertion.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                num_gpu_blocks_override=4,
                router="0.0.0.0:30000",
                kv_cache_ratio=1,
            )

        # Patch EngineWorkerQueue before EngineService ctor to avoid real IPC
        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

            def num_tasks(self):
                return 0

            def num_cache_infos(self):
                return 0

            def disaggregate_queue_empty(self):
                return True

            def get_disaggregated_tasks(self):
                return []

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # Patch heavy pieces
        eng.create_data_processor = lambda: setattr(eng, "data_processor", self._stub_processor())
        eng._process_splitwise_task = lambda: None
        eng._schedule_request_to_worker = lambda: None
        eng._schedule_request_to_worker_v1 = lambda: None

        started_cache = {}

        def fake_start_cache(device_ids, suffix):
            started_cache["called"] = True
            # return a list to mimic processes
            return [object()]

        eng.start_cache_service = fake_start_cache

        # Signals: make loaded_model_signal ready immediately; include launched_cache_manager_signal
        class Sig:
            def __init__(self, v=0):
                self.value = np.array([v], dtype=np.int32)

            def clear(self):
                pass

        def fake_init_signals():
            eng.worker_ready_signal = Sig(0)
            eng.loaded_model_signal = Sig(1)  # ready -> skip wait loop
            eng.launched_cache_manager_signal = Sig(0)

        eng._init_worker_signals = fake_init_signals

        # Worker start stub and initialization status -> False to trigger error path
        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
        eng.check_worker_initialize_status = lambda: False

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            # Avoid starting token processor loop
            eng.token_processor.run = lambda: None
            ok = eng.start(async_llm_pid=12345)

        # start() returns False on failure
        self.assertFalse(ok)
        # cache manager started before workers (lines 184-185)
        self.assertTrue(started_cache.get("called", False))
        # launched_cache_manager_signal set (line 221)
        self.assertEqual(int(eng.launched_cache_manager_signal.value[0]), 1)
        # avoid atexit finalizer
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_start_mixed_branch_cache_after_load_and_zmq(self):
        """Cover lines 215-217 and 231 in start()."""
        cfg = self._make_cfg(splitwise_role="mixed", num_gpu_blocks_override=4)

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

            def num_tasks(self):
                return 0

            def num_cache_infos(self):
                return 0

            def disaggregate_queue_empty(self):
                return True

            def get_disaggregated_tasks(self):
                return []

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        eng.create_data_processor = lambda: setattr(eng, "data_processor", self._stub_processor())
        eng._process_splitwise_task = lambda: None
        eng._schedule_request_to_worker = lambda: None
        eng._schedule_request_to_worker_v1 = lambda: None

        started_cache = {}

        def fake_start_cache(device_ids, suffix):
            started_cache["called"] = True
            return [object()]

        eng.start_cache_service = fake_start_cache

        class Sig:
            def __init__(self, v=0):
                self.value = np.array([v], dtype=np.int32)

            def clear(self):
                pass

        def fake_init_signals():
            eng.worker_ready_signal = Sig(0)
            eng.loaded_model_signal = Sig(1)
            eng.launched_cache_manager_signal = Sig(0)

        eng._init_worker_signals = fake_init_signals

        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: None)
        eng.check_worker_initialize_status = lambda: True

        zmq_called = {}
        eng.start_zmq_service = lambda pid: zmq_called.setdefault("pid", pid)

        with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
            eng.token_processor.run = lambda: None
            eng.start(async_llm_pid=8888)

        self.assertTrue(started_cache.get("called", False))  # lines 215-217
        self.assertEqual(zmq_called.get("pid"), 8888)  # line 231
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_insert_zmq_task_error_logging(self):
        """Cover lines 934-935 and 937 in _insert_zmq_task_to_scheduler."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=False)
        eng.running = True

        class DummyRecv:
            def __init__(self, msg):
                self.msg = msg

            def receive_json_once(self, block):
                return self.msg, None

            def close(self):
                pass

        # Case 1: context terminated -> info branch
        eng.recv_request_server = DummyRecv("Context was terminated")
        with patch.object(eng, "llm_logger") as _:
            eng._insert_zmq_task_to_scheduler()

        # Case 2: other error -> error branch
        eng.recv_request_server = DummyRecv("Other Error")
        with patch.object(eng, "llm_logger") as _:
            eng._insert_zmq_task_to_scheduler()
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_exit_sub_services_cleanup_paths(self):
        """Cover lines 1312-1340, 1350-1354 in _exit_sub_services."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # attach stubs used by cleanup
        class Sig:
            def __init__(self):
                self.value = np.array([0], dtype=np.int32)

            def clear(self):
                pass

        eng.worker_ready_signal = Sig()
        eng.loaded_model_signal = Sig()
        eng.exist_task_signal = Sig()
        eng.exist_swapped_task_signal = Sig()
        eng.worker_healthy_live_signal = Sig()
        eng.cache_ready_signal = Sig()
        eng.swap_space_ready_signal = Sig()
        eng.exist_prefill_task_signal = Sig()
        eng.model_weights_status_signal = Sig()
        eng.prefix_tree_status_signal = Sig()
        eng.kv_cache_status_signal = Sig()
        eng.send_response_server = Mock()
        eng.recv_request_server = Mock()
        eng.recv_control_cmd_server = Mock()

        # ensure cache manager control flags exist before first call
        eng.resource_manager.cache_manager.shm_cache_task_flag_broadcast = Mock(clear=lambda: None)
        eng.resource_manager.cache_manager.cache_ready_signal = Mock(clear=lambda: None)
        eng.cache_manager_processes = []

        # worker_proc kill raises -> cover 1312-1313
        eng.worker_proc = MagicMock(pid=1001)
        with patch("fastdeploy.engine.common_engine.os.getpgid", side_effect=RuntimeError("boom")):
            eng._exit_sub_services()

        # Prepare cache manager processes to hit both normal and exception branch
        class DummyCacheMgr:
            def __init__(self, pid, raise_on_kill=False):
                self.pid = pid
                self.raise_on_kill = raise_on_kill

        eng.cache_manager_processes = [DummyCacheMgr(2001, False), DummyCacheMgr(2002, True)]
        eng.resource_manager.cache_manager.shm_cache_task_flag_broadcast = Mock(clear=lambda: None)
        eng.resource_manager.cache_manager.cache_ready_signal = Mock(clear=lambda: None)

        def fake_getpgid(pid):
            return pid

        def fake_killpg(pid, sig):
            if pid == 2002:
                raise RuntimeError("kill fail")

        # cache_task_queue with cleanup
        eng.cache_task_queue = Mock()
        eng.cache_task_queue.cleanup = Mock()

        eng.dp_processed = [Mock(pid=3001, join=lambda: None)]
        eng.dp_engine_worker_queue_server = [Mock(cleanup=lambda: None)]

        with (
            patch("fastdeploy.engine.common_engine.os.getpgid", side_effect=fake_getpgid),
            patch("fastdeploy.engine.common_engine.os.killpg", side_effect=fake_killpg),
        ):
            eng._exit_sub_services()

        # Now cover manager.shutdown warning path (no cleanup attribute)
        class DummyMgr:
            def __init__(self):
                self.manager = Mock(shutdown=Mock(side_effect=RuntimeError("shutdown fail")))

        eng.cache_task_queue = DummyMgr()
        eng._exit_sub_services()
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_setting_environ_variables_v1_prefill_mm(self):
        """Cover lines 1476-1485 in _setting_environ_variables."""
        # For prefill + local scheduler the core code now requires a router
        # and ENABLE_V1_KVCACHE_SCHEDULER=0 when using the default IPC protocol.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(splitwise_role="prefill", router="0.0.0.0:30000")
        cfg.model_config.enable_mm = True

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        with patch("fastdeploy.engine.common_engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):
            prefix = eng._setting_environ_variables()
        self.assertIn("FLAGS_use_pd_disaggregation_per_chunk=1", prefix)
        self.assertIn("FLAGS_fmt_write_cache_completed_signal=1", prefix)
        self.assertIn("FLAGS_max_partition_size=1024", prefix)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_start_worker_service_cmd_build(self):
        """Cover 1517, 1526, 1568, 1592, 1595 by building the worker command with mocks."""
        with patch("fastdeploy.config.get_host_ip", return_value="127.0.0.1"):
            cfg = self._make_cfg(
                splitwise_role="mixed", num_gpu_blocks_override=4, ips=["127.0.0.1", "127.0.0.2"], data_parallel_size=2
            )
        # Make model multi-modal so env var branch already covered above; here not required
        cfg.structured_outputs_config.logits_processors = ["A", "B"]

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.data_processor = self._stub_processor()

        captured = {"cmd": None}

        class DummyProc:
            def __init__(self):
                self.stdout = None

            def poll(self):
                return None

        def fake_popen(cmd, stdout, shell, preexec_fn):
            captured["cmd"] = cmd
            return DummyProc()

        with patch("fastdeploy.engine.common_engine.subprocess.Popen", side_effect=fake_popen):
            with patch("fastdeploy.engine.common_engine.llm_logger"):
                p = eng._start_worker_service()

        self.assertIsNotNone(p)
        self.assertIsInstance(captured["cmd"], str)
        # logits processors added (1568)
        self.assertIn("--logits-processors A B", captured["cmd"])  # type: ignore
        # num_gpu_blocks_override added (1592)
        self.assertIn("--num_gpu_blocks_override 4", captured["cmd"])  # type: ignore
        # ips/nnodes added when nnode > 1 (1595)
        self.assertIn("--nnodes 2", captured["cmd"])  # type: ignore
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_check_health_unhealthy(self):
        """Cover line 1628: unhealthy worker."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class Sig:
            def __init__(self, v):
                self.value = np.array([v], dtype=np.int32)

        # set worker live time far past threshold
        eng.worker_healthy_live_signal = Sig(int(time.time()) - 1000)
        ok, msg = eng.check_health(time_interval_threashold=1)
        self.assertFalse(ok)
        self.assertIn("Not Healthy".lower(), msg.lower())
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_launch_components_expert_parallel(self):
        """Cover 1635-1638, 1660-1676, 1684-1703 in launch_components()."""
        # For prefill + local scheduler the core code now requires a router
        # and ENABLE_V1_KVCACHE_SCHEDULER=0 when using the default IPC protocol.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                # enable expert parallel and dp > 1 to go into the branch
                data_parallel_size=2,
                enable_expert_parallel=True,
                router="0.0.0.0:30000",
            )

        # Provide EngineWorkerQueue stub for ctor
        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

            def get_server_port(self):
                return 0

            def cleanup(self):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=True, use_async_llm=True)

        # Init signals to create launched_expert_service_signal
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_MULTI_API_SERVER", False):
            eng.ipc_signal_suffix = cfg.parallel_config.engine_worker_queue_port[0]
            eng._init_worker_signals()

            # Don't create real queues/processes
            with (
                patch("fastdeploy.engine.common_engine.EngineWorkerQueue") as FakeQ,
                patch("fastdeploy.engine.common_engine.multiprocessing.Process") as FakeP,
            ):
                # Fake queue instances with cleanup
                FakeQ.return_value = Mock(cleanup=lambda: None)

                # When starting process, immediately mark the signal as 1 to break waiting loop
                def start_side_effect(*args, **kwargs):
                    # set value for dp id 1
                    eng.launched_expert_service_signal.value[1] = 1

                proc_instance = Mock(start=start_side_effect)
                FakeP.return_value = proc_instance

                # Avoid scheduler doing real work
                eng.scheduler.start = lambda *a, **k: None
                with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
                    eng.launch_components()

                # Verify expert service branch executed
                self.assertTrue(hasattr(eng, "dp_processed"))
                self.assertGreaterEqual(len(eng.dp_processed), 1)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_check_worker_initialize_status_progress(self):
        """Cover 1710-1762 by simulating stdout and ready signals."""
        cfg = self._make_cfg(splitwise_role="mixed")

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        # Fake worker process stdout content that matches regexes
        lines = [
            b"Loading checkpoint shards: 1\n",
            b"Start load layer 5\n",
        ]

        class DummyProc:
            def __init__(self, it):
                self._it = iter(it)

            @property
            def stdout(self):
                return self._it

            def poll(self):
                return None

        eng.worker_proc = DummyProc(lines)
        eng.worker_init_status = {}
        eng.cfg.model_config.num_hidden_layers = 8

        # worker_ready_signal makes _worker_processes_ready() return True
        class Sig:
            def __init__(self):
                self.value = np.array([1], dtype=np.int32)

        eng.worker_ready_signal = Sig()

        # Replace tqdm and sleep for fast execution
        class DummyPbar:
            def __init__(self):
                self.n = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, delta=0, *args, **kwargs):
                try:
                    self.n += int(delta)
                except Exception:
                    self.n = 0

            def refresh(self):
                pass

        with patch("fastdeploy.engine.common_engine.tqdm", lambda *a, **k: DummyPbar()):
            with patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None):
                ok = eng.check_worker_initialize_status()
        self.assertTrue(ok)
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_worker_processes_ready_false(self):
        """Cover line 1382 returning False."""
        cfg = self._make_cfg()

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)

        class Sig:
            def __init__(self):
                # less than worker_num_per_node
                self.value = np.array([0], dtype=np.int32)

        eng.worker_ready_signal = Sig()
        self.assertFalse(eng._worker_processes_ready())
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_init_worker_signals_profile_iluvatar(self):
        """Cover line 1434 by forcing iluvatar custom device and do_profile=True."""
        # do_profile=True when num_gpu_blocks_override is None
        cfg = self._make_cfg(num_gpu_blocks_override=None)

        class DummyQ:
            def __init__(self, *a, **k):
                pass

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        eng.ipc_signal_suffix = cfg.parallel_config.engine_worker_queue_port[0]
        with patch("fastdeploy.engine.common_engine.paddle.is_compiled_with_custom_device", return_value=True):
            eng._init_worker_signals()
        # signal should exist
        self.assertTrue(hasattr(eng, "get_profile_block_num_signal"))
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    def test_launch_components_dp_mode(self):
        """Cover 1648-1652 branch for DP scheduler mode."""
        # When ENABLE_V1_KVCACHE_SCHEDULER=1 the IPC cache-transfer protocol
        # is no longer supported; force it to 0 here to avoid the
        # NotImplementedError raised in EngineArgs.__post_init__ so we can
        # still exercise the DP branch of launch_components.
        with patch("fastdeploy.engine.args_utils.envs.ENABLE_V1_KVCACHE_SCHEDULER", 0):
            cfg = self._make_cfg(
                splitwise_role="prefill",
                data_parallel_size=2,
                scheduler_name="dp",
            )

        class DummyQ:
            def __init__(self, *a, **k):
                self.available_prefill_instances = type("X", (), {"put": lambda *_: None})()

        with patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
        # Patch scheduler.start so it doesn't do heavy work
        eng.scheduler.start = Mock()
        eng.launch_components()
        eng.scheduler.start.assert_called()
        if hasattr(eng, "_finalizer"):
            try:
                eng._finalizer.detach()
            except Exception:
                pass

    # ---------- New focused tests for remaining uncovered logic ----------

    def test_insert_tasks_prefill_success_and_error_paths(self):
        """Exercise insert_tasks success path and error recycling branch."""

        eng = self._make_minimal_engine()

        # resource manager behaviour
        eng.resource_manager.check_and_free_block_tables = Mock()
        eng.resource_manager.stop_flags = np.array([1, 1], dtype=np.int32)
        eng.resource_manager.allocate_resources_for_new_tasks = Mock(side_effect=lambda ts: ts)
        eng.resource_manager.available_batch = Mock(return_value=2)
        eng.resource_manager.available_block_num = Mock(return_value=8)
        eng.engine_worker_queue.put_tasks = Mock()

        # splitwise prefill role with one ok and one failed task
        eng.cfg.scheduler_config.splitwise_role = "prefill"
        ok_req = self._make_request("ok_1", 4)
        ok_req.trace_carrier = None
        ok_req.disaggregate_info = None
        ok_req.metrics.scheduler_recv_req_time = time.time()

        bad_req = self._make_request("bad_1", 4)
        bad_req.trace_carrier = None
        bad_req.disaggregate_info = None
        bad_req.metrics.scheduler_recv_req_time = time.time()

        def fake_check_decode_allocated(task):
            return (task.request_id.startswith("ok"), "boom")

        eng.split_connector.check_decode_allocated = fake_check_decode_allocated
        eng.scheduler.put_results = Mock()

        # run
        ok = eng.insert_tasks([ok_req, bad_req], current_id=0)
        self.assertTrue(ok)
        # bad request should be removed and send error result
        eng.scheduler.put_results.assert_called()
        # tasks sent to engine queue
        eng.engine_worker_queue.put_tasks.assert_called_once()

    def test_insert_tasks_raises_engine_error_when_no_resources(self):
        """Exercise EngineError branch when allocate_resources_for_new_tasks returns empty list."""
        from fastdeploy.utils import EngineError

        eng = self._make_minimal_engine()
        eng.resource_manager.check_and_free_block_tables = Mock()
        eng.resource_manager.stop_flags = np.array([1], dtype=np.int32)
        eng.resource_manager.allocate_resources_for_new_tasks = Mock(return_value=[])
        # Ensure check_decode_allocated returns a tuple for unpacking
        eng.split_connector.check_decode_allocated = Mock(return_value=(True, ""))

        req = self._make_request("nores_1", 8)
        req.trace_carrier = None
        req.disaggregate_info = None
        req.metrics.scheduler_recv_req_time = time.time()

        with self.assertRaises(EngineError):
            eng.insert_tasks([req])

    def test_insert_prefilled_requests_error_and_success(self):
        """Cover _insert_prefilled_requests branches for error_code != 200 and eos-handling."""
        from fastdeploy.engine.request import RequestOutput

        eng = self._make_minimal_engine()
        eng.resource_manager.req_dict = {}
        eng.resource_manager.tasks_list = []
        eng.resource_manager.stop_flags = []
        eng.resource_manager._recycle_block_tables = Mock()
        eng.scheduler.put_results = Mock()
        eng.token_processor.tokens_counter = {}

        # Prepare slot 0 mapping
        req = self._make_request("r1", 4)
        # Ensure prompt_token_ids is properly initialized as a list
        req.prompt_token_ids = [1, 2] if req.prompt_token_ids is None else req.prompt_token_ids
        req.metrics.decode_recv_req_time = time.time()
        req.metrics.decode_preallocate_req_time = time.time()
        eng.resource_manager.req_dict["r1"] = 0
        eng.resource_manager.tasks_list.append(req)
        eng.resource_manager.stop_flags.append(False)

        # 1) error result -> should recycle and send result
        err_out = RequestOutput(request_id="r1", finished=True, error_code=500, error_msg="xx")
        # Ensure outputs namespace exists (engine accesses outputs.token_ids).
        # Use a non-empty token_ids so that when internal adapter is disabled
        # the engine code will recycle based on error_code and not raise IndexError.
        err_out.outputs = types.SimpleNamespace(token_ids=[0])
        # Ensure metrics exists (engine accesses metrics fields)
        err_out.metrics = types.SimpleNamespace(
            decode_recv_req_time=None, decode_preallocate_req_time=None, decode_inference_start_time=None
        )
        ok = eng._insert_prefilled_requests([err_out])
        self.assertTrue(ok)
        eng.scheduler.put_results.assert_called()

        # 2) success with empty token_ids under internal adapter -> recycle and skip
        from fastdeploy.utils import envs

        with patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", True):
            eng.resource_manager.req_dict["r2"] = 1
            eng.resource_manager.tasks_list.append(self._make_request("r2", 2))
            eng.resource_manager.stop_flags.append(False)

            out = RequestOutput(request_id="r2", finished=True, error_code=200, error_msg="")
            out.outputs = types.SimpleNamespace(token_ids=[])
            out.metrics = types.SimpleNamespace(
                decode_recv_req_time=None, decode_preallocate_req_time=None, decode_inference_start_time=None
            )
            ok = eng._insert_prefilled_requests([out])
            self.assertTrue(ok)

    def test_update_requests_chunk_size(self):
        """Cover chunked prefill path in update_requests_chunk_size."""
        eng = self._make_minimal_engine()
        eng.cfg.cache_config.enable_chunked_prefill = True
        eng.cfg.scheduler_config.max_num_batched_tokens = 8
        eng.cfg.cache_config.block_size = 2
        eng.partial_chunked_tokens = [0, 4, 4, 4]

        r1 = self._make_request("r1", 6)
        r2 = self._make_request("r2", 2)
        eng.update_requests_chunk_size([r1, r2])
        self.assertTrue(hasattr(r1, "prefill_chunk_info"))
        self.assertTrue(hasattr(r2, "prefill_chunk_info"))
        self.assertGreater(len(r1.prefill_chunk_info), 0)

    def test_decode_token_with_text(self):
        """Cover _decode_token branch when returning text."""
        eng = self._make_minimal_engine()
        eng.data_processor = Mock()
        eng.data_processor.decode_status = {"rid": (0, 2)}
        eng.data_processor.ids2tokens = Mock(return_value=("hi", [10, 11, 12], None))
        with patch.object(envs, "FD_ENABLE_RETURN_TEXT", True):
            delta, token_ids = eng._decode_token([1, 2], "rid", True)
        self.assertEqual(delta, "hi")
        self.assertEqual(token_ids, [10, 11])
        self.assertNotIn("rid", eng.data_processor.decode_status)

    def test_zmq_send_generated_tokens_internal_adapter(self):
        """Cover _zmq_send_generated_tokens internal-adapter branch."""
        eng = self._make_minimal_engine()
        eng.running = True
        eng._decode_token = Mock(return_value=("txt", [1]))
        eng.send_response_server = Mock(send_response=Mock())
        content = RequestOutput(
            request_id="r1",
            outputs=types.SimpleNamespace(token_ids=[1], decode_type=0, text=""),
            finished=True,
        )

        def fake_get_results():
            # stop after one iteration
            eng.running = False
            return [[content]]

        eng.scheduler.get_results = fake_get_results
        with (
            patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()
        eng._decode_token.assert_called_once()
        eng.send_response_server.send_response.assert_called()

    def test_zmq_send_generated_tokens_ipc(self):
        """Cover _zmq_send_generated_tokens IPC branch."""
        eng = self._make_minimal_engine()
        eng.running = True
        eng._decode_token = Mock(return_value=("", [2]))
        eng.send_response_server = Mock(send_response=Mock())
        content = RequestOutput(
            request_id="r2",
            outputs=types.SimpleNamespace(token_ids=[2], decode_type=0, text=""),
            finished=True,
        )

        def fake_get_results():
            eng.running = False
            return {"r2": [content]}

        eng.scheduler.get_results = fake_get_results
        with (
            patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", False),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
        ):
            eng._zmq_send_generated_tokens()
        eng._decode_token.assert_called_once()
        eng.send_response_server.send_response.assert_called_once()

    def test_start_worker_queue_service_tcp(self):
        """Cover start_worker_queue_service with TCP sockets."""
        cfg = self._make_cfg()

        class DummyQ:
            def __init__(self, address=None, *a, **k):
                DummyQ.created.append(address)
                self._port = address[1] if isinstance(address, tuple) else 0

            def get_server_port(self):
                return self._port

            def cleanup(self):
                pass

            def num_cache_infos(self):
                return 0

            def exist_tasks(self):
                return False

        DummyQ.created = []

        with (
            patch.object(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False),
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue", DummyQ),
            patch("fastdeploy.engine.common_engine.EngineCacheQueue", DummyQ),
        ):
            eng = EngineService(cfg, start_queue=False, use_async_llm=True)
            eng.start_worker_queue_service(start_queue=True)
        self.assertGreaterEqual(len(DummyQ.created), 1)
        self.assertTrue(hasattr(eng, "engine_worker_queue"))

    def test_clear_data(self):
        """Cover clear_data happy path."""
        eng = self._make_minimal_engine()
        eng.send_response_server = Mock(req_dict={"a": 1}, close=lambda: None)
        eng.recv_request_server = Mock(req_dict={"b": 2}, close=lambda: None)
        eng.token_processor.clear_data = Mock()
        eng.engine_worker_queue.clear_data = Mock()
        ok = eng.clear_data()
        self.assertTrue(ok)

    def test_register_to_router(self):
        """Cover _register_to_router success path with synchronous thread stub."""
        eng = self._make_minimal_engine()

        class _RouterCfg:
            def __init__(self):
                self.api_server_host = "127.0.0.1"
                self.api_server_port = 9999
                self.router = "http://127.0.0.1:8080"

        eng.cfg.router_config = _RouterCfg()
        eng.cfg.register_info = {"x": 1}

        with (
            patch("fastdeploy.engine.common_engine.check_service_health", return_value=True),
            patch("fastdeploy.engine.common_engine.requests.post", return_value=types.SimpleNamespace(ok=True)),
            patch("fastdeploy.engine.common_engine.time.sleep", lambda *_: None),
            patch("fastdeploy.engine.common_engine.threading.Thread") as FakeThread,
        ):

            def _start_side_effect():
                target = FakeThread.call_args.kwargs["target"]
                target()

            FakeThread.return_value = types.SimpleNamespace(start=_start_side_effect)
            eng._register_to_router()
        # success should exit loop; nothing to assert besides no exception
