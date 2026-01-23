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

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = SimpleNamespace(enable_torch_proxy=lambda scope: None)

import fastdeploy.engine.engine as engine_module
import fastdeploy.entrypoints.engine_client as engine_client_module
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import EngineError, ParameterError, envs


class DummySignal:
    def __init__(self, name, array, dtype, suffix, create):
        self.name = name
        self.value = array
        self.dtype = dtype
        self.suffix = suffix
        self.create = create
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyMetrics:
    def __init__(self):
        self.scheduler_recv_req_time = 0
        self.preprocess_start_time = 0
        self.preprocess_end_time = 0


class FakeRequest:
    def __init__(self, data):
        self._data = data
        self.guided_json = data.get("guided_json")
        self.guided_regex = data.get("guided_regex")
        self.guided_choice = data.get("guided_choice")
        self.structural_tag = data.get("structural_tag")
        self.guided_grammar = data.get("guided_grammar")
        self.guided_json_object = data.get("guided_json_object")
        self.prompt_token_ids = data.get("prompt_token_ids", [])
        self.prompt_token_ids_len = data.get("prompt_token_ids_len", 0)
        self.need_prefill_tokens = data.get("need_prefill_tokens", 0)
        self.metrics = DummyMetrics()
        self.sampling_params = None

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value


class DummyScheduler:
    def __init__(self):
        self.requests = []
        self.started = []

    def put_requests(self, requests):
        self.requests.extend(requests)

    def start(self, *args):
        self.started.append(args)

    def get_results(self):
        return "results"


class DummyTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or ["a", "b"]

    def get_vocab(self):
        return {"</think>": 2, "<|IMAGE_PLACEHOLDER|>": 3, "\n": 4}


class DummyDataProcessor:
    def __init__(self, prompt_token_ids=None):
        self.prompt_token_ids = prompt_token_ids or [1, 2]
        self.tokenizer = DummyTokenizer()
        self.eos_token_id_len = 1
        self.pad_token_id = 0

    def process_request(self, request, max_model_len, **kwargs):
        request.prompt_token_ids = list(self.prompt_token_ids)
        request.prompt_token_ids_len = len(self.prompt_token_ids)
        return request

    def process_response(self, result):
        return result


class DummyClientTokenizer:
    def __init__(self, vocab=None, sp_model=None):
        self.vocab = vocab or ["a", "b", "c"]
        if sp_model is not None:
            self.sp_model = sp_model


class DummyClientProcessor:
    def __init__(self, prompt_token_ids=None, tokenizer=None):
        self.prompt_token_ids = prompt_token_ids or [1, 2]
        self.tokenizer = tokenizer or DummyClientTokenizer()

    def process_request_dict(self, task, max_model_len):
        task["prompt_token_ids"] = list(self.prompt_token_ids)


class AsyncDummyClientProcessor(DummyClientProcessor):
    async def process_request_dict(self, task, max_model_len):
        task["prompt_token_ids"] = list(self.prompt_token_ids)


class DummyProcess:
    def __init__(self, pid=123):
        self.pid = pid
        self.join_called = False
        self._polled = False
        self.started = False

    def join(self):
        self.join_called = True

    def poll(self):
        return None if not self._polled else 1

    def start(self):
        self.started = True


class DummyQueueServer:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class DummyZmqServer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def print(self):
        return None


class JsonConfig:
    def to_json_string(self):
        return "{}"


class DummyIPCSignal:
    def __init__(self, name=None, array=None, dtype=None, suffix=None, create=None, shm_size=None):
        self.name = name
        self.value = array if array is not None else [0]
        self.dtype = dtype
        self.suffix = suffix
        self.create = create
        if shm_size is not None:
            self.shm = SimpleNamespace(buf=bytearray(shm_size))


class DummyFileLock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyDealerConnectionManager:
    def __init__(self, pid, max_connections):
        self.pid = pid
        self.max_connections = max_connections

    async def get_connection(self, request_id):
        queue = asyncio.Queue()
        queue.put_nowait(([b"ok"],))
        return SimpleNamespace(write=lambda data: None), queue


class DummyZmqClient:
    def __init__(self, model, mode):
        self.model = model
        self.mode = mode
        self.connected = False
        self.sent_json = []
        self.sent_pyobj = []

    def connect(self):
        self.connected = True

    def send_json(self, payload):
        self.sent_json.append(payload)

    def send_pyobj(self, payload):
        self.sent_pyobj.append(payload)


def build_cfg():
    return DummyConfig(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            block_size=8,
            enc_dec_block_num=0,
            gpu_memory_utilization=0.9,
            kv_cache_ratio=0.5,
            num_cpu_blocks=1,
            max_encoder_cache=0,
            cache_transfer_protocol="tcp",
            total_block_num=10,
            reset=lambda num: None,
        ),
        parallel_config=SimpleNamespace(
            device_ids="0",
            engine_worker_queue_port=[1234, 1235],
            tensor_parallel_size=1,
            expert_parallel_size=1,
            chunked_moe_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_chunked_moe=False,
            disable_custom_all_reduce=False,
            use_internode_ll_two_stage=False,
            disable_sequence_parallel_moe=False,
            shutdown_comm_group_if_worker_idle=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            splitwise_role="prefill",
            name="splitwise",
        ),
        model_config=SimpleNamespace(
            max_model_len=10,
            model="demo",
            quantization={},
            runner="default",
            convert="none",
            override_pooler_config="",
            logprobs_mode="",
            max_logprobs=0,
            model_impl="",
            enable_logprob=False,
            lm_head_fp32=False,
            enable_entropy=False,
            num_hidden_layers=4,
        ),
        structured_outputs_config=SimpleNamespace(
            guided_decoding_backend="",
            reasoning_parser="",
            disable_any_whitespace=False,
            logits_processors=None,
        ),
        load_config=SimpleNamespace(load_strategy="", dynamic_load_weight=False, load_choices="", rsync_config={}),
        speculative_config=JsonConfig(),
        graph_opt_config=JsonConfig(),
        early_stop_config=JsonConfig(),
        plas_attention_config=JsonConfig(),
        eplb_config=JsonConfig(),
        routing_replay_config=JsonConfig(),
        master_ip="127.0.0.1",
        host_ip="127.0.0.1",
        register_info={},
        node_rank=0,
        worker_num_per_node=1,
        nnode=1,
        ips=["127.0.0.1", "127.0.0.2"],
    )


def build_fd_config(
    *,
    tensor_parallel_size=2,
    tensor_parallel_rank=0,
    local_data_parallel_id=0,
    node_rank=0,
    enable_eplb=False,
    splitwise_role="mixed",
    enable_logprob=False,
    enable_mm=False,
    enable_prefix_caching=False,
    swap_space=False,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            local_data_parallel_id=local_data_parallel_id,
        ),
        model_config=SimpleNamespace(
            enable_mm=enable_mm,
            enable_logprob=enable_logprob,
            max_model_len=8,
            num_hidden_layers=2,
            moe_num_experts=2,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            max_processor_cache=0,
            swap_space=swap_space,
            kvcache_storage_backend=None,
        ),
        scheduler_config=SimpleNamespace(splitwise_role=splitwise_role),
        structured_outputs_config=SimpleNamespace(reasoning_parser=""),
        limit_mm_per_prompt=0,
        mm_processor_kwargs={},
        tool_parser=None,
        node_rank=node_rank,
        eplb_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            redundant_expert_ip_shm_size=64,
            redundant_expert_api_user="user",
            redundant_expert_api_password="pass",
            redundant_expert_meta_dir="/tmp",
        ),
    )


def setup_client_metrics(monkeypatch):
    monkeypatch.setattr(engine_client_module.tracing, "trace_slice_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_client_module.tracing, "trace_slice_end", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_client_module.tracing, "trace_get_proc_propagate_context", lambda *args: {})
    monkeypatch.setattr(engine_client_module, "trace_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine_client_module,
        "main_process_metrics",
        SimpleNamespace(
            request_params_max_tokens=SimpleNamespace(observe=lambda value: None),
            prompt_tokens_total=SimpleNamespace(inc=lambda value: None),
            request_prompt_tokens=SimpleNamespace(observe=lambda value: None),
        ),
    )


def build_engine(cfg, data_processor=None):
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.guided_decoding_checker = None
    engine.engine = SimpleNamespace(
        data_processor=data_processor or DummyDataProcessor(),
        scheduler=DummyScheduler(),
    )
    return engine


class DummyResult:
    def __init__(self, finished=False):
        self.finished = finished

    def to_dict(self):
        return {"outputs": {"text": "ok", "reasoning_content": "trace"}}


def test_init_sets_do_profile_and_uses_paddle(monkeypatch):
    tensor = paddle.to_tensor([1, 2, 3])
    assert int(tensor.sum()) == 6

    cfg = DummyConfig(cache_config=SimpleNamespace(num_gpu_blocks_override=1), print=lambda: None)
    dummy_service = object()
    monkeypatch.setattr(engine_module, "EngineService", lambda cfg: dummy_service)
    monkeypatch.setattr(engine_module.main_process_metrics, "set_cache_config_info", lambda obj: None)
    monkeypatch.setattr(engine_module.tracing, "trace_set_thread_info", lambda name: None)

    engine = LLMEngine(cfg)

    assert engine.do_profile == 0
    assert engine.engine is dummy_service
    engine._finalizer.detach()


def test_add_requests_min_tokens_too_long(monkeypatch):
    cfg = build_cfg()
    cfg.model_config.max_model_len = 5
    request = FakeRequest({"request_id": "r1", "max_tokens": 2, "min_tokens": 1})
    engine = build_engine(cfg, data_processor=DummyDataProcessor([1, 2, 3, 4]))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r1", "min_tokens": 1, "max_tokens": 2})

    assert "Input text is too long" in str(exc.value)


def test_add_requests_input_ids_too_long(monkeypatch):
    cfg = build_cfg()
    cfg.model_config.max_model_len = 3
    request = FakeRequest({"request_id": "r2", "max_tokens": 2, "min_tokens": -5})
    engine = build_engine(cfg, data_processor=DummyDataProcessor([1, 2, 3, 4]))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r2", "min_tokens": -5, "max_tokens": 2})

    assert "exceeds the limit" in str(exc.value)


def test_add_requests_stop_sequences_and_temperature(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest(
        {
            "request_id": "r3",
            "max_tokens": 2,
            "min_tokens": 1,
            "stop_seqs_len": [1, 2],
        }
    )
    engine = build_engine(cfg)
    sampling_params = SamplingParams(max_tokens=2, min_tokens=1, temperature=0.0)

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)
    monkeypatch.setattr(envs, "FD_MAX_STOP_SEQS_NUM", 1, raising=False)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r3"}, sampling_params=sampling_params)

    assert sampling_params.temperature == pytest.approx(1e-06)
    assert "max_stop_seqs_num" in str(exc.value)


def test_add_requests_stop_sequence_length_limit(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest(
        {
            "request_id": "r4",
            "max_tokens": 2,
            "min_tokens": 1,
            "stop_seqs_len": [3],
        }
    )
    engine = build_engine(cfg)

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)
    monkeypatch.setattr(envs, "FD_STOP_SEQS_MAX_LEN", 1, raising=False)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r4"})

    assert "stop_seqs" in str(exc.value)


def test_add_requests_guided_backend_missing(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest({"request_id": "r5", "max_tokens": 2, "min_tokens": 1, "guided_json": {}})
    engine = build_engine(cfg)

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r5"})

    assert "guided_backend is None" in str(exc.value)


def test_add_requests_guided_checker_error(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest({"request_id": "r6", "max_tokens": 2, "min_tokens": 1, "guided_json": {}})
    engine = build_engine(cfg)
    engine.guided_decoding_checker = SimpleNamespace(schema_format=lambda req: (req, "bad schema"))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    with pytest.raises(EngineError) as exc:
        engine.add_requests({"request_id": "r6"})

    assert "bad schema" in str(exc.value)


def test_add_requests_guided_checker_success(monkeypatch):
    cfg = build_cfg()
    request = FakeRequest({"request_id": "r7", "max_tokens": 2, "min_tokens": 1, "guided_json": {}})
    engine = build_engine(cfg)
    engine.guided_decoding_checker = SimpleNamespace(schema_format=lambda req: (req, None))

    monkeypatch.setattr(engine_module.Request, "from_dict", lambda data: request)

    engine.add_requests({"request_id": "r7"})

    assert engine.engine.scheduler.requests == [request]
    assert request.prompt_token_ids_len == len(request.prompt_token_ids)


def test_worker_processes_ready():
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.worker_ready_signal = SimpleNamespace(value=[1])

    assert engine._worker_processes_ready() is True


def test_init_worker_signals_with_profile(monkeypatch):
    cfg = build_cfg()
    cfg.parallel_config.data_parallel_size = 2
    cfg.nnode = 1
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 1
    engine.ipc_signal_suffix = "123"

    monkeypatch.setattr(engine_module, "IPCSignal", DummySignal)
    monkeypatch.setattr(paddle, "is_compiled_with_custom_device", lambda name: True)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)

    engine._init_worker_signals()

    assert isinstance(engine.worker_ready_signal, DummySignal)
    assert isinstance(engine.launched_cache_manager_signal, DummySignal)
    assert isinstance(engine.launched_expert_service_signal, DummySignal)
    assert isinstance(engine.loaded_model_signal, DummySignal)
    assert isinstance(engine.get_profile_block_num_signal, DummySignal)
    assert len(engine.get_profile_block_num_signal.value) == cfg.worker_num_per_node


def test_exit_sub_services_cleans_resources(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg

    cache_manager = SimpleNamespace(
        shm_cache_task_flag_broadcast=DummySignal("shm", [0], None, "", True),
        cache_ready_signal=DummySignal("cache", [0], None, "", True),
    )
    engine.engine = SimpleNamespace(resource_manager=SimpleNamespace(cache_manager=cache_manager))
    engine.cache_manager_processes = [DummyProcess(pid=111)]
    engine.worker_ready_signal = DummySignal("worker", [0], None, "", True)
    engine.loaded_model_signal = DummySignal("loaded", [0], None, "", True)
    engine.get_profile_block_num_signal = DummySignal("profile", [0], None, "", True)
    engine.worker_proc = DummyProcess(pid=222)
    engine.zmq_server = DummyZmqServer()
    engine.dp_processed = [DummyProcess(pid=333)]
    engine.dp_engine_worker_queue_server = [DummyQueueServer()]

    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(OSError("fail")))

    engine._exit_sub_services()

    assert engine.worker_ready_signal.cleared is True
    assert engine.loaded_model_signal.cleared is True
    assert engine.get_profile_block_num_signal.cleared is True
    assert engine.zmq_server.closed is True
    assert engine.dp_processed[0].join_called is True
    assert engine.dp_engine_worker_queue_server[0].cleaned is True


def test_setting_environ_variables(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg

    monkeypatch.setattr(envs, "ENABLE_V1_KVCACHE_SCHEDULER", True, raising=False)

    prefix = engine._setting_environ_variables()

    assert "FLAGS_use_pd_disaggregation_per_chunk=1" in prefix
    assert "FLAGS_fmt_write_cache_completed_signal=1" in prefix


def test_start_worker_service_builds_command(monkeypatch):
    cfg = build_cfg()
    cfg.nnode = 2
    cfg.parallel_config.data_parallel_size = 1
    cfg.cache_config.num_gpu_blocks_override = 2
    cfg.structured_outputs_config.logits_processors = ["p1", "p2"]
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.engine = SimpleNamespace(data_processor=DummyDataProcessor())
    engine.data_processor = engine.engine.data_processor

    captured = {}

    def fake_popen(cmd, stdout, shell, preexec_fn):
        captured["cmd"] = cmd
        return "proc"

    monkeypatch.setattr(engine_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(engine_module.current_platform, "is_iluvatar", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    engine._start_worker_service()

    assert "--logits-processors p1 p2" in captured["cmd"]
    assert "--nnodes" in captured["cmd"]
    assert "--devices" not in captured["cmd"]


def test_format_and_add_data_builds_context(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    captured = {}

    def fake_add_requests(prompts):
        captured["prompts"] = prompts

    engine.add_requests = fake_add_requests

    prompts = {
        "context": [
            {"role": "system", "utterance": "sys"},
            {"role": "user", "utterance": "hi"},
            {"role": "assistant", "utterance": "yo"},
        ]
    }

    req_id = engine._format_and_add_data(prompts)

    assert prompts["system"] == "sys"
    assert prompts["prompt"] == ["hi", "yo"]
    assert prompts["max_tokens"] == cfg.model_config.max_model_len
    assert captured["prompts"]["request_id"] == req_id


def test_generate_streaming_and_completion():
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.data_processor = DummyDataProcessor()
    engine.engine = SimpleNamespace(
        data_processor=engine.data_processor,
        check_and_free_block_tables=lambda: None,
    )

    engine._format_and_add_data = lambda prompts: "req"
    engine._get_generated_tokens = lambda req_id: [DummyResult(False), DummyResult(True)]

    outputs = list(engine.generate({"prompt": "hi"}, stream=True))

    assert outputs[0]["outputs"]["text"] == "ok"
    assert outputs[-1]["outputs"]["text"] == ""


def test_get_generated_result_returns_scheduler_output():
    engine = LLMEngine.__new__(LLMEngine)
    scheduler = DummyScheduler()
    scheduler.get_results = lambda: {"result": "ok"}
    engine.engine = SimpleNamespace(scheduler=scheduler)

    assert engine._get_generated_result() == {"result": "ok"}


def test_generate_wraps_add_request_errors():
    engine = LLMEngine.__new__(LLMEngine)
    engine._format_and_add_data = lambda prompts: (_ for _ in ()).throw(ValueError("bad"))

    with pytest.raises(EngineError) as exc:
        list(engine.generate({"prompt": "hi"}, stream=False))

    assert "bad" in str(exc.value)


def test_stop_profile_resets_cache(monkeypatch):
    cfg = build_cfg()
    cfg.cache_config.enable_prefix_caching = True
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 1
    engine.ipc_signal_suffix = "suffix"
    engine.get_profile_block_num_signal = SimpleNamespace(value=[2])
    engine.engine = SimpleNamespace(
        resource_manager=SimpleNamespace(reset_cache_config=lambda cache: None),
        start_cache_service=lambda device_ids, suffix: "cache",
    )

    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)

    engine._stop_profile()

    assert engine.do_profile == 0


def test_check_health_detects_unhealthy_worker(monkeypatch):
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine = SimpleNamespace(worker_healthy_live_signal=SimpleNamespace(value=[100]))

    monkeypatch.setattr(engine_module.time, "time", lambda: 200)

    ok, message = engine.check_health(time_interval_threashold=30)

    assert ok is False
    assert message == "Worker Service Not Healthy"


def test_launch_components_dp_path(monkeypatch):
    cfg = build_cfg()
    cfg.scheduler_config.name = "dp"
    cfg.parallel_config.data_parallel_size = 2
    cfg.nnode = 1
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.engine_worker_queue_port = [1000, 1001]
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=DummyScheduler(),
    )
    engine.launched_expert_service_signal = SimpleNamespace(value=[1, 1])

    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())

    class DummyContext:
        def Process(self, target, args):
            return DummyProcess(pid=500)

    monkeypatch.setattr(engine_module.multiprocessing, "get_context", lambda name: DummyContext())
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    engine.launch_components()

    assert engine.dp_engine_worker_queue_server
    assert engine.dp_processed


def test_check_worker_initialize_status(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.worker_init_status = {}  # Initialize the attribute

    stdout_lines = [
        b"Loading checkpoint shards: 50",
        b"Start load layer 2",
    ]

    engine.worker_proc = SimpleNamespace(stdout=stdout_lines, poll=lambda: None)
    engine._worker_processes_ready = lambda: True

    class DummyTqdm:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, delta):
            self.n += delta

        def refresh(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", DummyTqdm)
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    assert engine.check_worker_initialize_status() is True
    assert engine.worker_init_status["finished"] is True


def test_worker_processes_not_ready(monkeypatch):
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    # Set worker_ready_signal to have fewer ready workers than expected
    engine.worker_ready_signal = SimpleNamespace(value=[1, 0])  # Only 1 out of 2 workers ready
    engine.cfg.worker_num_per_node = 2

    assert engine._worker_processes_ready() is False


def test_check_health_worker_healthy(monkeypatch):
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine = SimpleNamespace(worker_healthy_live_signal=SimpleNamespace(value=[int(time.time())]))

    healthy, message = engine.check_health(time_interval_threashold=30)

    assert healthy is True
    assert message == ""


def test_launch_non_mixed_mode_starts_cache_manager(monkeypatch):
    """Test that cache manager starts in non-mixed mode for non-HPU platforms."""
    cfg = build_cfg()
    cfg.scheduler_config.splitwise_role = "prefill"  # Not mixed
    cfg.parallel_config.device_ids = "0,1"
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.ipc_signal_suffix = "test"
    engine.is_started = False

    # Mock cache manager processes
    mock_cache_processes = [DummyProcess(pid=123)]
    mock_engine = SimpleNamespace(
        start_cache_service=lambda device_ids, suffix: mock_cache_processes,
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=DummyDataProcessor(),
    )

    monkeypatch.setattr(engine_module, "current_platform", SimpleNamespace(is_intel_hpu=lambda: False))
    monkeypatch.setattr(engine, "engine", mock_engine, raising=False)
    monkeypatch.setattr(engine, "_start_worker_service", lambda: DummyProcess(pid=456))
    monkeypatch.setattr(engine, "_init_worker_signals", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_workers_ready", lambda: None, raising=False)
    monkeypatch.setattr(engine, "launch_components", lambda: None)
    monkeypatch.setattr(engine_module.time, "sleep", lambda x: None)

    engine.loaded_model_signal = SimpleNamespace(value=[1])
    engine.check_worker_initialize_status = lambda: True

    engine.start()

    assert engine.cache_manager_processes == mock_cache_processes


def test_launch_mixed_mode_starts_cache_manager_after_profile(monkeypatch):
    """Test that cache manager starts in mixed mode after profiling."""
    cfg = build_cfg()
    cfg.scheduler_config.splitwise_role = "mixed"
    cfg.cache_config.enable_prefix_caching = True
    cfg.parallel_config.device_ids = "0,1"
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.ipc_signal_suffix = "test"
    engine.is_started = False

    # Mock signals
    engine.loaded_model_signal = SimpleNamespace(value=[1])
    engine.launched_expert_service_signal = SimpleNamespace(value=[1])
    engine.worker_ready_signal = SimpleNamespace(value=[1])

    # Mock cache manager processes
    mock_cache_processes = [DummyProcess(pid=789)]
    mock_engine = SimpleNamespace(
        start_cache_service=lambda device_ids, suffix: mock_cache_processes,
        scheduler=SimpleNamespace(start=lambda *args: None),
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=DummyDataProcessor(),
    )

    monkeypatch.setattr(engine_module, "current_platform", SimpleNamespace(is_intel_hpu=lambda: False))
    monkeypatch.setattr(engine, "engine", mock_engine, raising=False)
    monkeypatch.setattr(engine, "_start_worker_service", lambda: DummyProcess(pid=456))
    monkeypatch.setattr(engine, "_init_worker_signals", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_workers_ready", lambda: None, raising=False)
    monkeypatch.setattr(
        engine,
        "_stop_profile",
        lambda: setattr(engine, "cache_manager_processes", mock_cache_processes),
    )
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())
    monkeypatch.setattr(
        engine_module.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=lambda *args, **kwargs: DummyProcess(pid=500)),
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(engine, "launch_components", lambda: None)

    engine.loaded_model_signal = SimpleNamespace(value=[1])
    engine.check_worker_initialize_status = lambda: True

    engine.start()

    assert engine.cache_manager_processes == mock_cache_processes


def test_launch_non_mixed_mode_sets_cache_manager_signal(monkeypatch):
    """Test that cache manager signal remains unset when cache manager is skipped."""
    cfg = build_cfg()
    cfg.scheduler_config.splitwise_role = "prefill"  # Not mixed
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.do_profile = 0
    engine.ipc_signal_suffix = "test"
    engine.is_started = False

    # Mock signals
    engine.launched_cache_manager_signal = SimpleNamespace(value=[0])

    mock_engine = SimpleNamespace(
        start_cache_service=lambda device_ids, suffix: [],
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=DummyDataProcessor(),
    )

    monkeypatch.setattr(
        engine_module, "current_platform", SimpleNamespace(is_intel_hpu=lambda: True)
    )  # Skip cache manager
    monkeypatch.setattr(engine, "engine", mock_engine, raising=False)
    monkeypatch.setattr(engine, "_start_worker_service", lambda: DummyProcess(pid=456))
    monkeypatch.setattr(engine, "_init_worker_signals", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_workers_ready", lambda: None, raising=False)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())
    monkeypatch.setattr(
        engine_module.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=lambda *args, **kwargs: DummyProcess(pid=500)),
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(engine, "launch_components", lambda: None)
    monkeypatch.setattr(engine, "launch_components", lambda: None)

    engine.loaded_model_signal = SimpleNamespace(value=[1])
    engine.check_worker_initialize_status = lambda: True

    engine.start()

    assert engine.launched_cache_manager_signal.value[0] == 0


def test_worker_init_check_failure_path(monkeypatch):
    """Test worker initialization check failure path."""
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.worker_init_status = {}

    # Create a mock process that will "fail" during polling
    class FailingProcess:
        def __init__(self):
            self.stdout = [b"Loading checkpoint shards: 50"]
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return 1 if self.poll_count > 2 else None  # Fail after a few polls

    engine.worker_proc = FailingProcess()
    engine._worker_processes_ready = lambda: False

    class DummyTqdm:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, delta):
            self.n += delta

        def refresh(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", DummyTqdm)
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    result = engine.check_worker_initialize_status()

    assert result is False


def test_generate_processes_stream_results(monkeypatch):
    """Test generate method processes streaming results."""
    cfg = build_cfg()
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.data_processor = DummyDataProcessor()
    engine.engine = SimpleNamespace(
        data_processor=engine.data_processor,
        check_and_free_block_tables=lambda: None,
    )

    engine._format_and_add_data = lambda prompts: "req"
    engine._get_generated_tokens = lambda req_id: [
        DummyResult(False),  # Streaming result
        DummyResult(True),  # Final result
    ]

    # Mock process_response to return None for streaming, result for final
    def mock_process_response(result):
        if not result.finished:
            return None  # Skip streaming results
        return result

    monkeypatch.setattr(engine.data_processor, "process_response", mock_process_response)

    outputs = list(engine.generate({"prompt": "hi"}, stream=True))

    # Should only get the final result since streaming returns None
    assert len(outputs) == 1
    assert outputs[0]["outputs"]["text"] == ""


def test_launch_components_logs_tensor_parallel_info(monkeypatch):
    """Test that launch_components logs tensor parallel information."""
    cfg = build_cfg()
    cfg.scheduler_config.name = "dp"
    cfg.parallel_config.data_parallel_size = 2
    cfg.nnode = 1
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.engine_worker_queue_port = [1000, 1001]
    engine = LLMEngine.__new__(LLMEngine)
    engine.cfg = cfg
    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=DummyScheduler(),
    )
    engine.launched_expert_service_signal = SimpleNamespace(value=[1, 1])

    # Mock logging to capture messages
    logged_messages = []

    def mock_info(msg):
        logged_messages.append(msg)

    monkeypatch.setattr(engine_module.llm_logger, "info", mock_info)
    monkeypatch.setattr(envs, "FD_ENABLE_MULTI_API_SERVER", False, raising=False)
    monkeypatch.setattr(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False, raising=False)
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: DummyQueueServer())
    monkeypatch.setattr(engine_module, "start_data_parallel_service", lambda *args: None)
    monkeypatch.setattr(engine_module.multiprocessing, "Queue", lambda: object())
    monkeypatch.setattr(
        engine_module.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=lambda *args, **kwargs: DummyProcess(pid=500)),
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)

    engine.launch_components()

    # Check that tensor parallel info was logged
    tensor_parallel_logs = [msg for msg in logged_messages if "Engine is initialized successfully" in msg]
    assert len(tensor_parallel_logs) > 0


def test_engine_client_init_sets_parallel_info(monkeypatch):
    cfg = build_fd_config(tensor_parallel_size=2, local_data_parallel_id=1, node_rank=1, swap_space=True)

    class DummyInputPreprocessor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def create_processor(self):
            tokenizer = DummyClientTokenizer(sp_model=["a", "b", "c", "d"])
            return DummyClientProcessor(tokenizer=tokenizer)

    monkeypatch.setattr(engine_client_module, "InputPreprocessor", DummyInputPreprocessor)
    monkeypatch.setattr(engine_client_module, "IPCSignal", DummyIPCSignal)
    monkeypatch.setattr(engine_client_module, "DealerConnectionManager", DummyDealerConnectionManager)
    monkeypatch.setattr(engine_client_module, "FileLock", DummyFileLock)
    monkeypatch.setattr(engine_client_module.current_platform, "is_iluvatar", lambda: False)

    client = engine_client_module.EngineClient(pid=1, port=1234, fd_config=cfg)

    assert client.is_master is True
    assert client.data_parallel_info["dp_rank"] == 5
    assert client.enable_cache_transfer is True


def test_engine_client_init_eplb_signals(monkeypatch):
    cfg = build_fd_config(enable_eplb=True, tensor_parallel_size=2, tensor_parallel_rank=0)
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.fd_config = cfg
    client.tensor_parallel_size = cfg.parallel_config.tensor_parallel_size

    monkeypatch.setattr(engine_client_module, "IPCSignal", DummyIPCSignal)

    client.init_eplb_signals(ipc_signal_suffix="port")

    assert len(client.signal_clear_experts_token_stats_list) == 2
    assert client.rearrange_experts_signal.name == "rearrange_experts_status"
    assert hasattr(client.shm_rearrange_experts_ips_list, "shm")


def test_engine_client_create_zmq_client(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    monkeypatch.setattr(engine_client_module, "ZmqIpcClient", DummyZmqClient)

    client.create_zmq_client(model="m", mode="mode")

    assert client.zmq_client.connected is True


def test_engine_client_format_and_add_data_sets_request_id(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    captured = {}

    async def fake_add_requests(request):
        request["prompt_token_ids"] = [1, 2, 3]
        captured["request"] = request

    client.add_requests = fake_add_requests
    request = {"prompt": "hi"}

    result = asyncio.run(client.format_and_add_data(request))

    assert result == [1, 2, 3]
    assert captured["request"]["max_tokens"] == 5
    assert "request_id" in captured["request"]


def test_engine_client_add_requests_sends_child_tasks(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 8
    client.enable_logprob = False
    client.max_logprobs = 5
    client.ori_vocab_size = 10
    client.enable_prefix_caching = False
    client.enable_mm = False
    client.data_processor = DummyClientProcessor(prompt_token_ids=[1, 2])
    client.zmq_client = DummyZmqClient("m", "mode")
    sent = []

    def fake_send_task(task):
        sent.append(task["request_id"])

    client._send_task = fake_send_task

    monkeypatch.setattr(engine_client_module.tracing, "trace_slice_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_client_module.tracing, "trace_slice_end", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_client_module.tracing, "trace_get_proc_propagate_context", lambda *args: {})
    monkeypatch.setattr(engine_client_module, "trace_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine_client_module,
        "main_process_metrics",
        SimpleNamespace(
            request_params_max_tokens=SimpleNamespace(observe=lambda value: None),
            prompt_tokens_total=SimpleNamespace(inc=lambda value: None),
            request_prompt_tokens=SimpleNamespace(observe=lambda value: None),
        ),
    )

    metrics = {"preprocess_start_time": 0, "preprocess_end_time": 0}
    task = {
        "request_id": "req_1",
        "metrics": metrics,
        "max_tokens": 4,
        "min_tokens": 1,
        "n": 2,
    }

    asyncio.run(client.add_requests(task))

    assert task["prompt_token_ids_len"] == 2
    assert sent == ["req_2", "req_3"]


def test_engine_client_send_task_modes(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.zmq_client = DummyZmqClient("m", "mode")

    client.enable_mm = False
    monkeypatch.setattr(envs, "ENABLE_V1_DATA_PROCESSOR", False, raising=False)
    client._send_task({"request_id": "r"})

    assert client.zmq_client.sent_json == [{"request_id": "r"}]

    client.enable_mm = True
    monkeypatch.setattr(envs, "FD_ENABLE_E2W_TENSOR_CONVERT", False, raising=False)
    client._send_task({"request_id": "r2"})

    assert client.zmq_client.sent_pyobj == [{"request_id": "r2"}]


def test_engine_client_valid_parameters_errors():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    client.enable_logprob = False
    client.max_logprobs = 2
    client.ori_vocab_size = 4
    client.enable_prefix_caching = False

    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "r1", "max_tokens": 2, "logprobs": True})


def test_engine_client_check_health_unhealthy():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.worker_healthy_live_signal = SimpleNamespace(value=[0])
    client.worker_healthy_live_signal.value[0] = time.time() - 100

    ok, message = client.check_health(time_interval_threashold=30)

    assert ok is False
    assert message == "Worker Service Not Healthy"


def test_engine_client_run_control_method_timeout(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.zmq_client = DummyZmqClient("m", "mode")

    class TimeoutQueue:
        async def get(self):
            raise asyncio.TimeoutError

    class TimeoutConnectionManager:
        async def get_connection(self, request_id):
            return SimpleNamespace(write=lambda data: None), TimeoutQueue()

    client.connection_manager = TimeoutConnectionManager()

    class DummyControlRequest:
        def __init__(self):
            self.request_id = "req"

        def to_dict(self):
            return {"request_id": self.request_id}

    request = DummyControlRequest()

    response = asyncio.run(client.run_control_method(request))

    assert response.error_code == 500
    assert "Timeout" in response.error_message


def test_engine_client_is_workers_alive():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.NORMAL])

    ok, message = client.is_workers_alive()

    assert ok is True
    assert message == ""


def test_engine_client_update_and_clear_model_weight(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.enable_prefix_caching = True
    client.enable_cache_transfer = True
    client.data_parallel_info = {"dp_rank": 0, "local_dp_rank": 0}
    client.prefix_tree_status_signal = SimpleNamespace(value=[engine_client_module.PrefixTreeStatus.CLEARED])
    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.CLEARED])
    client.kv_cache_status_signal = SimpleNamespace(value=[engine_client_module.KVCacheStatus.CLEARED])
    client.clear_update_lock = DummyFileLock("path")

    sleep_calls = {"count": 0}

    def fake_sleep(_):
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            client.prefix_tree_status_signal.value[0] = engine_client_module.PrefixTreeStatus.NORMAL
        else:
            client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.NORMAL
            client.kv_cache_status_signal.value[0] = engine_client_module.KVCacheStatus.NORMAL

    monkeypatch.setattr(engine_client_module.time, "sleep", fake_sleep)

    status, payload = client.update_model_weight(timeout=2)

    assert status == 200
    assert payload["msg"] == "update model weight successfully"

    client.prefix_tree_status_signal.value[0] = engine_client_module.PrefixTreeStatus.NORMAL
    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.NORMAL
    client.kv_cache_status_signal.value[0] = engine_client_module.KVCacheStatus.NORMAL

    clear_sleep_calls = {"count": 0}

    def fake_sleep_clear(_):
        clear_sleep_calls["count"] += 1
        if clear_sleep_calls["count"] == 1:
            client.prefix_tree_status_signal.value[0] = engine_client_module.PrefixTreeStatus.CLEARED
        else:
            client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.CLEARED
            client.kv_cache_status_signal.value[0] = engine_client_module.KVCacheStatus.CLEARED

    monkeypatch.setattr(engine_client_module.time, "sleep", fake_sleep_clear)

    status, payload = client.clear_load_weight(timeout=2)

    assert status == 200
    assert payload["msg"] == "clear model weight successfully"


def test_engine_client_check_model_weight_status():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.model_weights_status_signal = SimpleNamespace(value=[-1])

    assert client.check_model_weight_status() is True


def test_engine_client_rearrange_experts_branches(monkeypatch):
    cfg = build_fd_config(enable_eplb=True, splitwise_role="prefill", tensor_parallel_rank=0)
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.fd_config = cfg
    client.rearrange_experts_signal = SimpleNamespace(value=[engine_client_module.RearrangeExpertStatus.FREE.value])
    client.rearrange_experts_ips_size_signal = SimpleNamespace(value=[0])
    client.shm_rearrange_experts_ips_list = DummyIPCSignal(
        name="ips", shm_size=cfg.eplb_config.redundant_expert_ip_shm_size
    )
    client.expert_tokens_stats_array_list = [
        SimpleNamespace(value=np.zeros((2, 2), dtype=np.int32)),
    ]
    client.signal_update_weight_from_disk_array_list = [SimpleNamespace(value=[0])]
    client.signal_update_weight_from_tensor_array = SimpleNamespace(value=[0])

    content, status = asyncio.run(client.rearrange_experts({"user": "user", "passwd": "pass", "ips": ["1.1.1.1:1"]}))

    assert status == engine_client_module.HTTPStatus.OK
    assert content["msg"] == "ok"

    content, status = asyncio.run(
        client.rearrange_experts(
            {"user": "user", "passwd": "pass", "action": "recv_expert_weight", "data": [[1, 2], [3, 4]]}
        )
    )

    assert status == engine_client_module.HTTPStatus.OK
    assert client.signal_update_weight_from_disk_array_list[0].value[0] == 1

    client.rearrange_experts_signal.value[0] = engine_client_module.RearrangeExpertStatus.LOAD_SUCC.value
    content, status = asyncio.run(
        client.rearrange_experts({"user": "user", "passwd": "pass", "action": "update_weight_from_tensor"})
    )

    assert status == engine_client_module.HTTPStatus.OK
    assert client.signal_update_weight_from_tensor_array.value[0] == 1


def test_engine_client_get_stats_and_check_redundant(monkeypatch):
    cfg = build_fd_config(enable_eplb=True, tensor_parallel_rank=0)
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.fd_config = cfg
    client.signal_clear_experts_token_stats_list = [SimpleNamespace(value=[0])]
    client.local_experts_token_stats_array_list = [
        SimpleNamespace(value=np.array([[1, 2], [3, 4]], dtype=np.int32)),
    ]
    client.update_weight_from_disk_result_list = [SimpleNamespace(value=np.array([1], dtype=np.int32))]
    client.rearrange_experts_signal = SimpleNamespace(value=[engine_client_module.RearrangeExpertStatus.FREE.value])

    content, status = asyncio.run(
        client.get_per_expert_tokens_stats({"user": "user", "passwd": "pass", "clear_stat": True})
    )

    assert status == engine_client_module.HTTPStatus.OK
    assert content["data"] == [[[1, 2], [3, 4]]]
    assert client.signal_clear_experts_token_stats_list[0].value[0] == 1

    monkeypatch.setattr(
        engine_client_module,
        "RedundantExpertWorkload",
        lambda path: SimpleNamespace(load=lambda: ({"work": 1}, "ok")),
    )

    content, status = asyncio.run(
        client.check_redundant({"user": "user", "passwd": "pass", "check_get_workloads": True})
    )

    assert status == engine_client_module.HTTPStatus.OK
    assert content["data"] == {"work": 1}

    content, status = asyncio.run(
        client.check_redundant({"user": "user", "passwd": "pass", "action": "check_load_weight_result"})
    )

    assert status == engine_client_module.HTTPStatus.OK
    assert content["data"] == [1]


def test_engine_client_abort_sends_requests(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    sent = []

    def fake_send_task(task):
        sent.append(task["request_id"])

    client._send_task = fake_send_task
    monkeypatch.setattr(envs, "FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True, raising=False)

    asyncio.run(client.abort("req_3", n=2))

    assert sent == ["req_0", "req_1"]


def test_engine_client_init_master_false_and_eplb_skip(monkeypatch):
    cfg = build_fd_config(enable_eplb=True, tensor_parallel_size=32, tensor_parallel_rank=1)

    class DummyInputPreprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def create_processor(self):
            return DummyClientProcessor()

    monkeypatch.setattr(engine_client_module, "InputPreprocessor", DummyInputPreprocessor)
    monkeypatch.setattr(engine_client_module, "IPCSignal", DummyIPCSignal)
    monkeypatch.setattr(engine_client_module, "DealerConnectionManager", DummyDealerConnectionManager)
    monkeypatch.setattr(engine_client_module, "FileLock", DummyFileLock)
    monkeypatch.setattr(engine_client_module.current_platform, "is_iluvatar", lambda: True)

    client = engine_client_module.EngineClient(pid=2, port=4321, fd_config=cfg)

    assert client.is_master is False


def test_engine_client_add_requests_async_and_direct_send(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    client.enable_logprob = False
    client.max_logprobs = 4
    client.ori_vocab_size = 8
    client.enable_prefix_caching = False
    client.enable_mm = False
    client.data_processor = AsyncDummyClientProcessor(prompt_token_ids=[1, 2])
    client.zmq_client = DummyZmqClient("m", "mode")
    sent = []

    def fake_send_task(task):
        sent.append(task["request_id"])

    client._send_task = fake_send_task
    setup_client_metrics(monkeypatch)

    metrics = {"preprocess_start_time": 0, "preprocess_end_time": 0}
    task = {
        "request_id": "req",
        "metrics": metrics,
        "max_tokens": 4,
        "min_tokens": 1,
        "messages": ["msg"],
    }

    asyncio.run(client.add_requests(task))

    assert task["messages"] is None
    assert sent == ["req"]


def test_engine_client_add_requests_processing_error(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    client.enable_logprob = False
    client.max_logprobs = 4
    client.ori_vocab_size = 8
    client.enable_prefix_caching = False
    client.enable_mm = False

    class FailingProcessor(DummyClientProcessor):
        def process_request_dict(self, task, max_model_len):
            raise ValueError("boom")

    client.data_processor = FailingProcessor()
    client.zmq_client = DummyZmqClient("m", "mode")
    client._send_task = lambda task: None
    setup_client_metrics(monkeypatch)

    with pytest.raises(EngineError):
        asyncio.run(client.add_requests({"request_id": "req", "metrics": {}}))


def test_engine_client_add_requests_length_errors(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 3
    client.enable_logprob = False
    client.max_logprobs = 4
    client.ori_vocab_size = 8
    client.enable_prefix_caching = False
    client.enable_mm = False
    client.data_processor = DummyClientProcessor(prompt_token_ids=[1, 2])
    client._send_task = lambda task: None
    setup_client_metrics(monkeypatch)

    with pytest.raises(EngineError):
        asyncio.run(client.add_requests({"request_id": "req", "metrics": {}, "max_tokens": 2, "min_tokens": 1}))

    client.data_processor = DummyClientProcessor(prompt_token_ids=[1, 2, 3, 4])

    with pytest.raises(EngineError):
        asyncio.run(client.add_requests({"request_id": "req", "metrics": {}, "max_tokens": 1}))


def test_engine_client_add_requests_stop_sequences(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 10
    client.enable_logprob = False
    client.max_logprobs = 4
    client.ori_vocab_size = 8
    client.enable_prefix_caching = False
    client.enable_mm = False
    client.data_processor = DummyClientProcessor(prompt_token_ids=[1])
    client._send_task = lambda task: None
    setup_client_metrics(monkeypatch)

    monkeypatch.setattr(envs, "FD_MAX_STOP_SEQS_NUM", 0, raising=False)

    with pytest.raises(EngineError):
        asyncio.run(client.add_requests({"request_id": "req", "metrics": {}, "max_tokens": 2, "stop_seqs_len": [1]}))

    monkeypatch.setattr(envs, "FD_MAX_STOP_SEQS_NUM", 2, raising=False)
    monkeypatch.setattr(envs, "FD_STOP_SEQS_MAX_LEN", 1, raising=False)

    with pytest.raises(EngineError):
        asyncio.run(client.add_requests({"request_id": "req", "metrics": {}, "max_tokens": 2, "stop_seqs_len": [2]}))


def test_engine_client_add_requests_send_error(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 10
    client.enable_logprob = False
    client.max_logprobs = 4
    client.ori_vocab_size = 8
    client.enable_prefix_caching = False
    client.enable_mm = False
    client.data_processor = DummyClientProcessor(prompt_token_ids=[1])

    def raise_send(task):
        raise ValueError("send error")

    client._send_task = raise_send
    setup_client_metrics(monkeypatch)

    with pytest.raises(EngineError):
        asyncio.run(client.add_requests({"request_id": "req", "metrics": {}, "max_tokens": 2}))


def test_engine_client_send_task_tensor_convert(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.zmq_client = DummyZmqClient("m", "mode")
    client.enable_mm = True
    monkeypatch.setattr(envs, "FD_ENABLE_E2W_TENSOR_CONVERT", True, raising=False)
    called = {"value": False}

    def fake_to_tensor(payload):
        called["value"] = True

    monkeypatch.setattr(engine_client_module, "to_tensor", fake_to_tensor)

    client._send_task({"request_id": "r3"})

    assert called["value"] is True
    assert client.zmq_client.sent_pyobj == [{"request_id": "r3"}]


def test_engine_client_valid_parameters_adjustments(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    client.enable_logprob = True
    client.max_logprobs = 4
    client.ori_vocab_size = 10
    client.enable_prefix_caching = False
    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", True, raising=False)

    data = {
        "request_id": "req",
        "max_tokens": 4,
        "reasoning_max_tokens": 5,
        "temperature": 0.0,
        "logprobs": 2,
    }

    client.valid_parameters(data)

    assert data["reasoning_max_tokens"] == 4
    assert data["temperature"] == pytest.approx(1e-6)


def test_engine_client_valid_parameters_max_tokens_error():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 4
    client.enable_logprob = True
    client.max_logprobs = 2
    client.ori_vocab_size = 4
    client.enable_prefix_caching = False

    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 0})


def test_engine_client_valid_parameters_reasoning_error():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 4
    client.enable_logprob = True
    client.max_logprobs = 2
    client.ori_vocab_size = 4
    client.enable_prefix_caching = False

    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "reasoning_max_tokens": 0})


def test_engine_client_valid_parameters_logprobs_invalid_type():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 4
    client.enable_logprob = True
    client.max_logprobs = 2
    client.ori_vocab_size = 4
    client.enable_prefix_caching = False

    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": "bad"})


def test_engine_client_valid_parameters_max_logprobs_bounds(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 4
    client.enable_logprob = True
    client.enable_prefix_caching = False
    client.ori_vocab_size = 5
    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", True, raising=False)

    client.max_logprobs = -1
    client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": 1})

    client.max_logprobs = -2
    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": 1})

    client.max_logprobs = 6
    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": 1})


def test_engine_client_valid_parameters_prompt_logprobs(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    client.enable_prefix_caching = False
    client.ori_vocab_size = 5
    client.max_logprobs = 2

    client.enable_logprob = False
    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "prompt_logprobs": 1})

    client.enable_logprob = True
    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", False, raising=False)
    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "prompt_logprobs": 1})

    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", True, raising=False)
    client.enable_prefix_caching = True
    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "prompt_logprobs": 1})

    client.enable_prefix_caching = False
    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "prompt_logprobs": -1})

    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "prompt_logprobs": -2})

    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "prompt_logprobs": 5})


def test_engine_client_valid_parameters_top_logprobs(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.max_model_len = 6
    client.ori_vocab_size = 6
    client.max_logprobs = 3
    client.enable_prefix_caching = False

    client.enable_logprob = False
    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": 1})

    client.enable_logprob = True
    with pytest.raises(ParameterError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": True, "top_logprobs": "bad"})

    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": 5})

    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", False, raising=False)
    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": -1})

    monkeypatch.setattr(envs, "FD_USE_GET_SAVE_OUTPUT_V1", True, raising=False)
    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": -1})

    with pytest.raises(ValueError):
        client.valid_parameters({"request_id": "req", "max_tokens": 2, "logprobs": -2})


def test_engine_client_check_health_healthy():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.worker_healthy_live_signal = SimpleNamespace(value=[0])

    ok, message = client.check_health(time_interval_threashold=30)

    assert ok is True
    assert message == ""


def test_engine_client_run_control_method_success(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.zmq_client = DummyZmqClient("m", "mode")

    class ResponseQueue:
        async def get(self):
            return [
                {
                    "request_id": "req",
                    "error_code": 200,
                    "error_message": None,
                    "result": {"ok": True},
                    "finished": True,
                }
            ]

    class SuccessConnectionManager:
        async def get_connection(self, request_id):
            return SimpleNamespace(write=lambda data: None), ResponseQueue()

    client.connection_manager = SuccessConnectionManager()

    class DummyControlRequest:
        def __init__(self):
            self.request_id = "req"

        def to_dict(self):
            return {"request_id": self.request_id}

    request = DummyControlRequest()

    response = asyncio.run(client.run_control_method(request))

    assert response.error_code == 200
    assert response.result == {"ok": True}


def test_engine_client_is_workers_alive_unavailable():
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.CLEARED])

    ok, message = client.is_workers_alive()

    assert ok is False
    assert message == "No model weight enabled"


def test_engine_client_update_model_weight_early_returns(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.enable_prefix_caching = False
    client.enable_cache_transfer = False
    client.data_parallel_info = {"dp_rank": 0, "local_dp_rank": 0}
    client.clear_update_lock = DummyFileLock("path")

    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.NORMAL])
    status, payload = client.update_model_weight(timeout=1)
    assert status == 200
    assert payload["msg"] == "model weight is updated"

    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.UPDATING
    status, payload = client.update_model_weight(timeout=1)
    assert status == 400

    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.CLEARING
    status, payload = client.update_model_weight(timeout=1)
    assert status == 403


def test_engine_client_update_model_weight_timeouts(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.enable_prefix_caching = True
    client.enable_cache_transfer = True
    client.data_parallel_info = {"dp_rank": 0, "local_dp_rank": 0}
    client.clear_update_lock = DummyFileLock("path")
    client.prefix_tree_status_signal = SimpleNamespace(value=[engine_client_module.PrefixTreeStatus.CLEARED])
    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.CLEARED])
    client.kv_cache_status_signal = SimpleNamespace(value=[engine_client_module.KVCacheStatus.CLEARED])

    monkeypatch.setattr(engine_client_module.time, "sleep", lambda _: None)

    status, payload = client.update_model_weight(timeout=0)
    assert status == 404
    assert payload["msg"] == "update prefix tree timeout"

    client.enable_prefix_caching = False
    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.CLEARED
    client.kv_cache_status_signal.value[0] = engine_client_module.KVCacheStatus.CLEARED

    status, payload = client.update_model_weight(timeout=0)
    assert status == 404
    assert payload["msg"] == "update model weight timeout"


def test_engine_client_clear_load_weight_early_returns(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.enable_prefix_caching = False
    client.enable_cache_transfer = False
    client.data_parallel_info = {"dp_rank": 0, "local_dp_rank": 0}
    client.clear_update_lock = DummyFileLock("path")
    client.kv_cache_status_signal = SimpleNamespace(value=[engine_client_module.KVCacheStatus.CLEARED])
    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.CLEARED])

    status, payload = client.clear_load_weight(timeout=1)
    assert status == 200

    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.CLEARING
    status, payload = client.clear_load_weight(timeout=1)
    assert status == 400

    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.UPDATING
    status, payload = client.clear_load_weight(timeout=1)
    assert status == 403


def test_engine_client_clear_load_weight_timeouts(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.enable_prefix_caching = True
    client.enable_cache_transfer = True
    client.data_parallel_info = {"dp_rank": 0, "local_dp_rank": 0}
    client.clear_update_lock = DummyFileLock("path")
    client.prefix_tree_status_signal = SimpleNamespace(value=[engine_client_module.PrefixTreeStatus.NORMAL])
    client.model_weights_status_signal = SimpleNamespace(value=[engine_client_module.ModelWeightsStatus.NORMAL])
    client.kv_cache_status_signal = SimpleNamespace(value=[engine_client_module.KVCacheStatus.CLEARED])

    monkeypatch.setattr(engine_client_module.time, "sleep", lambda _: None)

    status, payload = client.clear_load_weight(timeout=0)
    assert status == 404
    assert payload["msg"] == "clear prefix tree timeout"

    client.enable_prefix_caching = False
    client.model_weights_status_signal.value[0] = engine_client_module.ModelWeightsStatus.NORMAL
    client.kv_cache_status_signal.value[0] = engine_client_module.KVCacheStatus.CLEARED

    status, payload = client.clear_load_weight(timeout=0)
    assert status == 404
    assert payload["msg"] == "clear model weight timeout"


def test_engine_client_rearrange_experts_error_branches():
    cfg = build_fd_config(enable_eplb=False)
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.fd_config = cfg

    content, status = asyncio.run(client.rearrange_experts({"user": "x", "passwd": "y"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST
    assert content["msg"] == "redundant expert is disabled"

    cfg.eplb_config.enable_eplb = True
    content, status = asyncio.run(client.rearrange_experts({"user": "bad", "passwd": "y"}))
    assert status == engine_client_module.HTTPStatus.UNAUTHORIZED

    cfg.parallel_config.tensor_parallel_rank = 1
    content, status = asyncio.run(client.rearrange_experts({"user": "user", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    cfg.parallel_config.tensor_parallel_rank = 0
    cfg.eplb_config.redundant_expert_ip_shm_size = 1
    client.rearrange_experts_signal = SimpleNamespace(value=[engine_client_module.RearrangeExpertStatus.DOING.value])
    client.rearrange_experts_ips_size_signal = SimpleNamespace(value=[0])
    client.shm_rearrange_experts_ips_list = DummyIPCSignal(name="ips", shm_size=1)
    content, status = asyncio.run(client.rearrange_experts({"user": "user", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    client.rearrange_experts_signal.value[0] = engine_client_module.RearrangeExpertStatus.FREE.value
    content, status = asyncio.run(client.rearrange_experts({"user": "user", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    content, status = asyncio.run(client.rearrange_experts({"user": "user", "passwd": "pass", "ips": ["1.1.1.1:1"]}))
    assert status == engine_client_module.HTTPStatus.INTERNAL_SERVER_ERROR

    content, status = asyncio.run(
        client.rearrange_experts({"user": "user", "passwd": "pass", "action": "recv_expert_weight"})
    )
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    cfg.scheduler_config.splitwise_role = "mixed"
    content, status = asyncio.run(
        client.rearrange_experts({"user": "user", "passwd": "pass", "action": "update_weight_from_tensor"})
    )
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    cfg.scheduler_config.splitwise_role = "prefill"
    client.rearrange_experts_signal.value[0] = engine_client_module.RearrangeExpertStatus.FREE.value
    content, status = asyncio.run(
        client.rearrange_experts({"user": "user", "passwd": "pass", "action": "update_weight_from_tensor"})
    )
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    content, status = asyncio.run(client.rearrange_experts({"user": "user", "passwd": "pass", "action": "invalid"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST


def test_engine_client_get_per_expert_tokens_stats_errors():
    cfg = build_fd_config(enable_eplb=False)
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.fd_config = cfg

    content, status = asyncio.run(client.get_per_expert_tokens_stats({"user": "x", "passwd": "y"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    cfg.eplb_config.enable_eplb = True
    content, status = asyncio.run(client.get_per_expert_tokens_stats({"user": "bad", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.UNAUTHORIZED

    cfg.parallel_config.tensor_parallel_rank = 1
    content, status = asyncio.run(client.get_per_expert_tokens_stats({"user": "user", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST


def test_engine_client_check_redundant_errors_and_unknown_status(monkeypatch):
    cfg = build_fd_config(enable_eplb=False)
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client.fd_config = cfg

    content, status = asyncio.run(client.check_redundant({"user": "x", "passwd": "y"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    cfg.eplb_config.enable_eplb = True
    content, status = asyncio.run(client.check_redundant({"user": "bad", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.UNAUTHORIZED

    cfg.parallel_config.tensor_parallel_rank = 1
    content, status = asyncio.run(client.check_redundant({"user": "user", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.BAD_REQUEST

    cfg.parallel_config.tensor_parallel_rank = 0
    client.rearrange_experts_signal = SimpleNamespace(value=[999])
    content, status = asyncio.run(client.check_redundant({"user": "user", "passwd": "pass"}))
    assert status == engine_client_module.HTTPStatus.OK
    assert content["status"] == "unknown"


def test_engine_client_abort_edge_cases(monkeypatch):
    client = engine_client_module.EngineClient.__new__(engine_client_module.EngineClient)
    client._send_task = lambda task: None
    monkeypatch.setattr(envs, "FD_ENABLE_REQUEST_DISCONNECT_STOP_INFERENCE", True, raising=False)

    asyncio.run(client.abort("req", n=0))
    asyncio.run(client.abort("req", n=1))
