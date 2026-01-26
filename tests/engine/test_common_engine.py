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

import time
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = SimpleNamespace(enable_torch_proxy=lambda scope=None: None)

from fastdeploy.engine import engine as engine_module
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import EngineError


class DummyMetrics:
    def __init__(self):
        self.scheduler_recv_req_time = None
        self.preprocess_start_time = None
        self.preprocess_end_time = None


class DummyRequest:
    def __init__(self, prompt_token_ids=None, sampling_params=None, **kwargs):
        self.prompt_token_ids = prompt_token_ids or []
        self.prompt_token_ids_len = None
        self.need_prefill_tokens = None
        self.metrics = DummyMetrics()
        self.sampling_params = sampling_params or SamplingParams()
        self.guided_json = kwargs.get("guided_json")
        self.guided_regex = kwargs.get("guided_regex")
        self.guided_choice = kwargs.get("guided_choice")
        self.structural_tag = kwargs.get("structural_tag")
        self.guided_grammar = kwargs.get("guided_grammar")
        self.guided_json_object = kwargs.get("guided_json_object")
        self.stop_seqs_len = kwargs.get("stop_seqs_len")
        self.request_id = kwargs.get("request_id", "req")
        self.chat_template = None

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self.sampling_params, key):
            return getattr(self.sampling_params, key)
        return default

    def set(self, key, value):
        if hasattr(self.sampling_params, key):
            setattr(self.sampling_params, key, value)
        else:
            setattr(self, key, value)


class DummySignal:
    def __init__(self, name, array, dtype, suffix, create):
        self.name = name
        self.value = array
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyTokenizer:
    def __init__(self):
        self.vocab = {"</think>": 10, "<|IMAGE_PLACEHOLDER|>": 11, "\n": 12}

    def get_vocab(self):
        return self.vocab


class DummyEngineArgs:
    def __init__(self, cfg):
        self._cfg = cfg

    def create_engine_config(self):
        return self._cfg


def build_cfg(splitwise_role="prefill"):
    cache_config = SimpleNamespace(
        gpu_memory_utilization=0.9,
        block_size=8,
        enc_dec_block_num=0,
        kv_cache_ratio=0.5,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        num_gpu_blocks_override=None,
        num_cpu_blocks=1,
        cache_transfer_protocol="tcp",
        max_encoder_cache=0,
        total_block_num=0,
    )
    cache_config.reset_called_with = None

    def reset(num_blocks):
        cache_config.reset_called_with = num_blocks

    cache_config.reset = reset
    model_config = SimpleNamespace(
        max_model_len=8,
        model="dummy",
        quantization={"bits": 4},
        num_hidden_layers=2,
        runner="runner",
        convert=False,
        override_pooler_config="",
        logprobs_mode="none",
        max_logprobs=0,
        model_impl="impl",
        enable_logprob=False,
        lm_head_fp32=False,
        enable_entropy=False,
    )
    scheduler_config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=32,
        splitwise_role=splitwise_role,
        name="splitwise",
    )
    parallel_config = SimpleNamespace(
        engine_worker_queue_port=[5555],
        device_ids="0",
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
    )
    structured_outputs_config = SimpleNamespace(
        guided_decoding_backend="none",
        logits_processors=None,
        disable_any_whitespace=False,
        reasoning_parser="",
    )
    load_config = SimpleNamespace(
        load_strategy="auto",
        rsync_config={},
        dynamic_load_weight=False,
        load_choices="",
    )
    to_json = lambda: "{}"
    cfg = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        structured_outputs_config=structured_outputs_config,
        load_config=load_config,
        speculative_config=SimpleNamespace(to_json_string=to_json),
        graph_opt_config=SimpleNamespace(to_json_string=to_json),
        early_stop_config=SimpleNamespace(to_json_string=to_json),
        eplb_config=SimpleNamespace(to_json_string=to_json),
        routing_replay_config=SimpleNamespace(to_json_string=to_json),
        plas_attention_config=SimpleNamespace(to_json_string=to_json),
        master_ip="127.0.0.1",
        ips=None,
        nnode=1,
        host_ip="127.0.0.1",
        register_info=None,
        node_rank=0,
        worker_num_per_node=1,
    )
    cfg.print = lambda: None
    return cfg


def test_has_guided_input_detection():
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    blank_request = DummyRequest()
    guided_request = DummyRequest(guided_json={"a": 1})
    assert engine._has_guided_input(blank_request) is False
    assert engine._has_guided_input(guided_request) is True


def test_from_engine_args_initializes_engine(monkeypatch):
    cfg = build_cfg()
    cfg.cache_config.num_gpu_blocks_override = None
    created = {}

    class DummyEngineService:
        def __init__(self, passed_cfg):
            created["cfg"] = passed_cfg

    monkeypatch.setattr(engine_module, "EngineService", DummyEngineService)
    monkeypatch.setattr(engine_module.weakref, "finalize", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        engine_module.main_process_metrics, "set_cache_config_info", lambda obj: created.setdefault("metrics", obj)
    )
    monkeypatch.setattr(engine_module.tracing, "trace_set_thread_info", lambda name: created.setdefault("trace", name))

    engine = engine_module.LLMEngine.from_engine_args(DummyEngineArgs(cfg))

    assert engine.cfg is cfg
    assert created["cfg"] is cfg
    assert engine.do_profile == 1
    assert created["metrics"] is cfg.cache_config
    assert created["trace"] == "engine"


def test_from_engine_args_disables_profile(monkeypatch):
    cfg = build_cfg()
    cfg.cache_config.num_gpu_blocks_override = 1
    created = {}

    class DummyEngineService:
        def __init__(self, passed_cfg):
            created["cfg"] = passed_cfg

    monkeypatch.setattr(engine_module, "EngineService", DummyEngineService)
    monkeypatch.setattr(engine_module.weakref, "finalize", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(engine_module.main_process_metrics, "set_cache_config_info", lambda obj: None)
    monkeypatch.setattr(engine_module.tracing, "trace_set_thread_info", lambda name: None)

    engine = engine_module.LLMEngine.from_engine_args(DummyEngineArgs(cfg))

    assert engine.do_profile == 0
    assert created["cfg"] is cfg


def test_add_requests_updates_sampling_and_puts_request(monkeypatch):
    paddle.ones([1])
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.guided_decoding_checker = None
    captured = {}

    def fake_from_dict(task):
        return DummyRequest(prompt_token_ids=[1, 2], sampling_params=SamplingParams(min_tokens=1, max_tokens=6))

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1, 2]
        captured["chat_template_kwargs"] = kwargs.get("chat_template_kwargs")
        return request

    class DummyScheduler:
        def put_requests(self, requests):
            captured["requests"] = requests

    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=DummyScheduler(),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)
    sampling_params = SamplingParams(temperature=0.0, min_tokens=1, max_tokens=6)

    engine.add_requests({"prompt": "hi"}, sampling_params=sampling_params)

    assert sampling_params.temperature == pytest.approx(1e-06)
    assert captured["requests"][0].prompt_token_ids_len == 2
    assert captured["chat_template_kwargs"]["chat_template"] is None


def test_add_requests_raises_on_min_tokens_exceed(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.guided_decoding_checker = None

    def fake_from_dict(task):
        return DummyRequest(prompt_token_ids=[1, 2, 3, 4, 5], sampling_params=SamplingParams(min_tokens=4))

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        return request

    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=SimpleNamespace(put_requests=lambda requests: None),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)

    with pytest.raises(EngineError) as excinfo:
        engine.add_requests({"prompt": "hi"}, sampling_params=SamplingParams(min_tokens=4, max_tokens=6))
    assert excinfo.value.error_code == 400


def test_add_requests_raises_on_input_ids_exceed(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.guided_decoding_checker = None

    def fake_from_dict(task):
        return DummyRequest(
            prompt_token_ids=[1] * 10,
            sampling_params=SimpleNamespace(min_tokens=-5, max_tokens=6),
        )

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1] * 10
        return request

    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=SimpleNamespace(put_requests=lambda requests: None),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)

    with pytest.raises(EngineError):
        engine.add_requests({"prompt": "hi"}, sampling_params=None)


def test_add_requests_raises_on_stop_seqs_limits(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.guided_decoding_checker = None

    def fake_from_dict(task):
        return DummyRequest(prompt_token_ids=[1], sampling_params=SamplingParams(min_tokens=1))

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1]
        request.stop_seqs_len = [5, 5]
        return request

    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=SimpleNamespace(put_requests=lambda requests: None),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)
    monkeypatch.setattr(engine_module.envs, "FD_MAX_STOP_SEQS_NUM", 1)

    with pytest.raises(EngineError):
        engine.add_requests({"prompt": "hi"}, sampling_params=SamplingParams(min_tokens=1, max_tokens=6))


def test_add_requests_raises_on_stop_seqs_length(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.guided_decoding_checker = None

    def fake_from_dict(task):
        return DummyRequest(prompt_token_ids=[1], sampling_params=SamplingParams(min_tokens=1))

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1]
        request.stop_seqs_len = [10]
        return request

    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=SimpleNamespace(put_requests=lambda requests: None),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)
    monkeypatch.setattr(engine_module.envs, "FD_STOP_SEQS_MAX_LEN", 5)
    monkeypatch.setattr(engine_module.envs, "FD_MAX_STOP_SEQS_NUM", 10)

    with pytest.raises(EngineError):
        engine.add_requests({"prompt": "hi"}, sampling_params=SamplingParams(min_tokens=1, max_tokens=6))


def test_add_requests_guided_decoding_requires_backend(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.guided_decoding_checker = None

    def fake_from_dict(task):
        return DummyRequest(prompt_token_ids=[1], guided_json={"a": 1}, sampling_params=SamplingParams(min_tokens=1))

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1]
        return request

    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=SimpleNamespace(put_requests=lambda requests: None),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)

    with pytest.raises(EngineError):
        engine.add_requests({"prompt": "hi"}, sampling_params=SamplingParams(min_tokens=1, max_tokens=6))


def test_add_requests_guided_decoding_schema_error(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()

    def fake_from_dict(task):
        return DummyRequest(prompt_token_ids=[1], guided_json={"a": 1}, sampling_params=SamplingParams(min_tokens=1))

    def fake_process_request(request, max_model_len, **kwargs):
        request.prompt_token_ids = [1]
        return request

    engine.guided_decoding_checker = SimpleNamespace(schema_format=lambda request: (request, "bad schema"))
    engine.engine = SimpleNamespace(
        data_processor=SimpleNamespace(process_request=fake_process_request),
        scheduler=SimpleNamespace(put_requests=lambda requests: None),
    )
    monkeypatch.setattr(engine_module.Request, "from_dict", fake_from_dict)

    with pytest.raises(EngineError):
        engine.add_requests({"prompt": "hi"}, sampling_params=SamplingParams(min_tokens=1, max_tokens=6))


def test_worker_signals_and_ready(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.do_profile = True
    engine.ipc_signal_suffix = "5555"
    monkeypatch.setattr(engine_module, "IPCSignal", DummySignal)
    monkeypatch.setattr(paddle, "is_compiled_with_custom_device", lambda name: False)

    engine._init_worker_signals()
    engine.worker_ready_signal.value[:] = 1
    assert engine._worker_processes_ready() is True
    engine.worker_ready_signal.value[:] = 0
    assert engine._worker_processes_ready() is False


def test_init_worker_signals_with_dp_and_profile(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.cfg.parallel_config.data_parallel_size = 2
    engine.cfg.worker_num_per_node = 2
    engine.cfg.nnode = 1
    engine.do_profile = True
    engine.ipc_signal_suffix = "5555"
    monkeypatch.setattr(engine_module.envs, "FD_ENABLE_MULTI_API_SERVER", False)
    monkeypatch.setattr(engine_module, "IPCSignal", DummySignal)
    monkeypatch.setattr(paddle, "is_compiled_with_custom_device", lambda name: True)

    engine._init_worker_signals()

    assert engine.worker_ready_signal.value.shape[0] == 2
    assert engine.launched_expert_service_signal.value.shape[0] == 2
    assert engine.loaded_model_signal.value.shape[0] == 1
    assert engine.get_profile_block_num_signal.value.shape[0] == 2


def test_setting_environ_variables_includes_flags(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="prefill")
    monkeypatch.setattr(engine_module.envs, "ENABLE_V1_KVCACHE_SCHEDULER", True)
    command_prefix = engine._setting_environ_variables()
    assert "FLAGS_use_pd_disaggregation_per_chunk=1" in command_prefix
    assert "FLAGS_fmt_write_cache_completed_signal=1" in command_prefix


def test_setting_environ_variables_disaggregation_default(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="prefill")
    monkeypatch.setattr(engine_module.envs, "ENABLE_V1_KVCACHE_SCHEDULER", False)
    command_prefix = engine._setting_environ_variables()
    assert "FLAGS_use_pd_disaggregation=1" in command_prefix


def test_start_worker_service_builds_command(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.do_profile = 0
    tokenizer = DummyTokenizer()
    data_processor = SimpleNamespace(tokenizer=tokenizer, eos_token_id_len=1, pad_token_id=0)
    engine.engine = SimpleNamespace(data_processor=data_processor)
    engine.data_processor = data_processor
    engine._setting_environ_variables = lambda: "TEST_ENV=1"

    captured = {}

    def fake_popen(cmd, stdout, shell, preexec_fn):
        captured["cmd"] = cmd
        return SimpleNamespace(pid=1234, stdout=stdout)

    monkeypatch.setattr(engine_module.subprocess, "Popen", fake_popen)

    process = engine._start_worker_service()

    assert process.pid == 1234
    assert "--max_model_len 8" in captured["cmd"]
    assert "--engine_worker_queue_port 5555" in captured["cmd"]
    assert "--log_dir" in captured["cmd"]


def test_start_worker_service_no_think_token(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.do_profile = 0
    tokenizer = SimpleNamespace(
        vocab={"<|IMAGE_PLACEHOLDER|>": 11, "\n": 12},
        get_vocab=lambda: {"<|IMAGE_PLACEHOLDER|>": 11, "\n": 12},
    )
    data_processor = SimpleNamespace(tokenizer=tokenizer, eos_token_id_len=1, pad_token_id=0)
    engine.engine = SimpleNamespace(data_processor=data_processor)
    engine.data_processor = data_processor
    engine._setting_environ_variables = lambda: "TEST_ENV=1"

    captured = {}

    def fake_popen(cmd, stdout, shell, preexec_fn):
        captured["cmd"] = cmd
        return SimpleNamespace(pid=2222, stdout=stdout)

    monkeypatch.setattr(engine_module.subprocess, "Popen", fake_popen)

    process = engine._start_worker_service()

    assert process.pid == 2222
    assert "--image_patch_id 11" in captured["cmd"]


def test_start_worker_service_with_flags_and_iluvatar(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.cfg.parallel_config.enable_expert_parallel = True
    engine.cfg.parallel_config.enable_chunked_moe = True
    engine.cfg.cache_config.enable_chunked_prefill = True
    engine.cfg.load_config.dynamic_load_weight = True
    engine.cfg.structured_outputs_config.disable_any_whitespace = True
    engine.cfg.parallel_config.disable_custom_all_reduce = True
    engine.cfg.parallel_config.use_internode_ll_two_stage = True
    engine.cfg.parallel_config.disable_sequence_parallel_moe = True
    engine.cfg.model_config.enable_logprob = True
    engine.cfg.model_config.lm_head_fp32 = True
    engine.cfg.parallel_config.shutdown_comm_group_if_worker_idle = True
    engine.cfg.model_config.enable_entropy = True
    engine.cfg.cache_config.num_gpu_blocks_override = 4
    engine.cfg.structured_outputs_config.logits_processors = ["a", "b"]
    engine.cfg.ips = ["127.0.0.1", "127.0.0.2"]
    engine.cfg.nnode = 2
    engine.do_profile = 1
    tokenizer = DummyTokenizer()
    data_processor = SimpleNamespace(tokenizer=tokenizer, eos_token_id_len=1, pad_token_id=0)
    engine.engine = SimpleNamespace(data_processor=data_processor)
    engine.data_processor = data_processor
    engine._setting_environ_variables = lambda: "TEST_ENV=1"
    monkeypatch.setattr(engine_module.current_platform, "is_iluvatar", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    captured = {}

    def fake_popen(cmd, stdout, shell, preexec_fn):
        captured["cmd"] = cmd
        return SimpleNamespace(pid=4321, stdout=stdout)

    monkeypatch.setattr(engine_module.subprocess, "Popen", fake_popen)

    process = engine._start_worker_service()

    assert process.pid == 4321
    assert "--logits-processors a b" in captured["cmd"]
    assert "--num_gpu_blocks_override 4" in captured["cmd"]
    assert f"--devices {engine.cfg.parallel_config.device_ids}" not in captured["cmd"]


def test_format_and_add_data_sets_context(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    captured = {}

    def fake_add_requests(payload):
        captured["payload"] = payload

    engine.add_requests = fake_add_requests
    request_id = engine._format_and_add_data(
        {
            "context": [
                {"role": "system", "utterance": "system"},
                {"role": "user", "utterance": "hi"},
                {"role": "assistant", "utterance": "ok"},
            ]
        }
    )

    assert request_id == captured["payload"]["request_id"]
    assert captured["payload"]["system"] == "system"
    assert captured["payload"]["prompt"] == ["hi", "ok"]
    assert captured["payload"]["max_tokens"] == engine.cfg.model_config.max_model_len


def test_format_and_add_data_preserves_request_id(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    captured = {}

    def fake_add_requests(payload):
        captured["payload"] = payload

    engine.add_requests = fake_add_requests
    request_id = engine._format_and_add_data({"request_id": "fixed", "prompt": "hi"})

    assert request_id == "fixed"
    assert captured["payload"]["request_id"] == "fixed"


def test_generate_handles_streaming(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine._format_and_add_data = lambda prompts: "req"
    engine.engine = SimpleNamespace(check_and_free_block_tables=lambda: None)

    class DummyOutput:
        def __init__(self, text):
            self.text = text

        def to_dict(self):
            return {"outputs": {"text": self.text, "reasoning_content": self.text}}

    engine.engine.data_processor = SimpleNamespace(process_response=lambda result: DummyOutput(result.payload))

    class Result:
        def __init__(self, payload, finished):
            self.payload = payload
            self.finished = finished

    def fake_get_generated_tokens(req_id):
        return iter([Result("mid", False), Result("final", True)])

    engine._get_generated_tokens = fake_get_generated_tokens

    outputs = list(engine.generate({"prompt": "hi"}, stream=True))
    assert outputs[0]["outputs"]["text"] == "mid"
    assert outputs[1]["outputs"]["text"] == ""


def test_generate_streaming_skips_none_processed():
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine._format_and_add_data = lambda prompts: "req"
    engine.engine = SimpleNamespace(check_and_free_block_tables=lambda: None)

    engine.engine.data_processor = SimpleNamespace(
        process_response=lambda result: (
            None
            if result.payload == "skip"
            else SimpleNamespace(to_dict=lambda: {"outputs": {"text": "done", "reasoning_content": "done"}})
        )
    )

    class Result:
        def __init__(self, payload, finished):
            self.payload = payload
            self.finished = finished

    engine._get_generated_tokens = lambda req_id: iter([Result("skip", False), Result("done", True)])

    outputs = list(engine.generate({"prompt": "hi"}, stream=True))
    assert outputs[-1]["outputs"]["text"] == ""


def test_generate_non_streaming_and_none_processed(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine._format_and_add_data = lambda prompts: "req"
    engine.engine = SimpleNamespace(check_and_free_block_tables=lambda: None)

    class DummyOutput:
        def __init__(self, text):
            self.text = text

        def to_dict(self):
            return {"outputs": {"text": self.text, "reasoning_content": self.text}}

    def process_response(result):
        if result.payload == "skip":
            return None
        return DummyOutput(result.payload)

    engine.engine.data_processor = SimpleNamespace(process_response=process_response)

    class Result:
        def __init__(self, payload, finished):
            self.payload = payload
            self.finished = finished

    def fake_get_generated_tokens(req_id):
        return iter([Result("skip", False), Result("final", True)])

    engine._get_generated_tokens = fake_get_generated_tokens

    outputs = list(engine.generate({"prompt": "hi"}, stream=False))
    assert outputs[0]["outputs"]["text"] == "final"


def test_generate_handles_add_request_error():
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine._format_and_add_data = lambda prompts: (_ for _ in ()).throw(ValueError("bad"))

    with pytest.raises(EngineError) as excinfo:
        list(engine.generate({"prompt": "hi"}, stream=False))
    assert excinfo.value.error_code == 400


def test_stop_profile_resets_cache_config(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.do_profile = 1
    engine.ipc_signal_suffix = "5555"
    engine.get_profile_block_num_signal = SimpleNamespace(value=np.array([2], dtype=np.int32))
    engine.engine = SimpleNamespace(
        resource_manager=SimpleNamespace(reset_cache_config=lambda config: None),
        start_cache_service=lambda device_ids, suffix: ["cache"],
    )
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)

    engine._stop_profile()

    assert engine.do_profile == 0
    assert engine.cfg.cache_config.reset_called_with == 2


def test_stop_profile_waits_for_signal(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.do_profile = 1
    engine.ipc_signal_suffix = "5555"
    signal = SimpleNamespace(value=np.array([0], dtype=np.int32))
    engine.get_profile_block_num_signal = signal
    engine.engine = SimpleNamespace(
        resource_manager=SimpleNamespace(reset_cache_config=lambda config: None),
        start_cache_service=lambda device_ids, suffix: ["cache"],
    )
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)

    def fake_sleep(_):
        signal.value[0] = 3

    monkeypatch.setattr(engine_module.time, "sleep", fake_sleep)

    engine._stop_profile()
    assert engine.cfg.cache_config.reset_called_with == 3


def test_check_health_detects_stale_worker():
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.engine = SimpleNamespace(worker_healthy_live_signal=SimpleNamespace(value=np.array([0.0])))
    ok, message = engine.check_health()
    assert ok is True
    assert message == ""
    engine.engine.worker_healthy_live_signal.value[0] = time.time() - 40
    ok, message = engine.check_health(time_interval_threashold=30)
    assert ok is False
    assert message == "Worker Service Not Healthy"


def test_get_generated_result():
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.engine = SimpleNamespace(scheduler=SimpleNamespace(get_results=lambda: ["result"]))
    assert engine._get_generated_result() == ["result"]


def test_start_profile_flow(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.is_started = False
    engine.do_profile = 1

    def fake_init_worker_signals():
        engine.loaded_model_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    engine._init_worker_signals = fake_init_worker_signals
    engine.launch_components = lambda: None
    engine.engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: ["cache"],
        start_zmq_service=lambda pid: None,
    )
    engine._start_worker_service = lambda: SimpleNamespace(pid=1, stdout=None)
    engine.check_worker_initialize_status = lambda: True
    engine._stop_profile = lambda: setattr(engine, "profile_stopped", True)
    monkeypatch.setattr(engine_module.envs, "ENABLE_V1_KVCACHE_SCHEDULER", True)
    monkeypatch.setattr(engine_module.envs, "FD_ENABLE_INTERNAL_ADAPTER", True)
    monkeypatch.setattr(engine_module.envs, "FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "1111,2222")
    monkeypatch.setattr(engine_module.envs, "FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "3333,4444")
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.start(api_server_pid=123) is True
    assert engine.profile_stopped is True
    assert engine_module.envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT == "1111"
    assert engine_module.envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT == "3333"


def test_start_returns_false_when_worker_init_fails(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.is_started = False
    engine.do_profile = 0

    def fake_init_worker_signals():
        engine.loaded_model_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    engine._init_worker_signals = fake_init_worker_signals
    engine.launch_components = lambda: None
    engine.engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: ["cache"],
    )
    engine._start_worker_service = lambda: SimpleNamespace(pid=1, stdout=None)
    engine.check_worker_initialize_status = lambda: False
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.start() is False


def test_start_waits_for_loaded_model(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.is_started = False
    engine.do_profile = 0
    engine.loaded_model_signal = SimpleNamespace(value=np.array([0], dtype=np.int32))
    done = {}

    def fake_init_worker_signals():
        engine.loaded_model_signal = SimpleNamespace(value=np.array([0], dtype=np.int32))

    def check_worker_initialize_status():
        while engine.loaded_model_signal.value[0] == 0:
            time.sleep(0.01)
        return True

    call_count = {"count": 0}

    def fake_sleep(_):
        call_count["count"] += 1
        if call_count["count"] > 1:
            engine.loaded_model_signal.value[0] = 1
        done["slept"] = True

    engine._init_worker_signals = fake_init_worker_signals
    engine.launch_components = lambda: None
    engine.engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: ["cache"],
    )
    engine._start_worker_service = lambda: SimpleNamespace(pid=1, stdout=None)
    engine.check_worker_initialize_status = check_worker_initialize_status
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)
    monkeypatch.setattr(engine_module.time, "sleep", fake_sleep)

    assert engine.start() is True
    assert done["slept"] is True


def test_start_cache_service_before_workers(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="prefill")
    engine.is_started = False
    engine.do_profile = 0

    def fake_init_worker_signals():
        engine.loaded_model_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    engine._init_worker_signals = fake_init_worker_signals
    engine.launch_components = lambda: None
    called = {}
    engine.engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: called.setdefault("cache", device_ids),
    )
    engine._start_worker_service = lambda: SimpleNamespace(pid=1, stdout=None)
    engine.check_worker_initialize_status = lambda: True
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.start() is True
    assert called["cache"] == ["0"]


def test_start_cache_service_mixed_after_profile(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="mixed")
    engine.is_started = False
    engine.do_profile = 0
    engine.cfg.cache_config.enable_prefix_caching = True

    def fake_init_worker_signals():
        engine.loaded_model_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    engine._init_worker_signals = fake_init_worker_signals
    engine.launch_components = lambda: None
    called = {}
    engine.engine = SimpleNamespace(
        start=lambda: None,
        create_data_processor=lambda: None,
        data_processor=SimpleNamespace(),
        start_cache_service=lambda device_ids, suffix: called.setdefault("cache", device_ids),
    )
    engine._start_worker_service = lambda: SimpleNamespace(pid=1, stdout=None)
    engine.check_worker_initialize_status = lambda: True
    monkeypatch.setattr(engine_module.current_platform, "is_intel_hpu", lambda: False)
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.start() is True
    assert called["cache"] == ["0"]


def test_launch_components_splitwise_and_dp(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="prefill")
    engine.cfg.scheduler_config.name = "dp"
    engine.cfg.parallel_config.data_parallel_size = 2
    engine.cfg.parallel_config.engine_worker_queue_port = [6000, 6001]
    engine.cfg.worker_num_per_node = 1
    engine.cfg.node_rank = 0
    engine.cfg.nnode = 1
    engine.cfg.master_ip = "127.0.0.1"
    engine.launched_expert_service_signal = SimpleNamespace(value=np.array([0, 1], dtype=np.int32))

    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=SimpleNamespace(start=lambda *args: None),
    )

    class DummyQueue:
        def __init__(self, *args, **kwargs):
            pass

    class DummyProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            pass

    class DummyContext:
        def Process(self, target, args):
            return DummyProcess(target, args)

    monkeypatch.setattr(engine_module.multiprocessing, "Queue", DummyQueue)
    monkeypatch.setattr(engine_module.multiprocessing, "get_context", lambda name: DummyContext())
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", lambda **kwargs: SimpleNamespace(cleanup=lambda: None))
    monkeypatch.setattr(engine_module.envs, "FD_ENABLE_MULTI_API_SERVER", False)
    monkeypatch.setattr(engine_module.envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False)
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    engine.launch_components()

    assert len(engine.dp_processed) == 1
    assert len(engine.dp_engine_worker_queue_server) == 1


def test_launch_components_splitwise_name(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="prefill")
    engine.cfg.scheduler_config.name = "splitwise"
    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=SimpleNamespace(start=lambda *args: args),
    )

    called = {}

    def start_scheduler(*args):
        called["args"] = args

    engine.engine.scheduler.start = start_scheduler
    monkeypatch.setattr(engine_module.envs, "FD_ENABLE_MULTI_API_SERVER", True)

    engine.launch_components()
    assert called["args"][0] == "prefill"


def test_launch_components_queue_shm(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg(splitwise_role="prefill")
    engine.cfg.scheduler_config.name = "dp"
    engine.cfg.parallel_config.data_parallel_size = 2
    engine.cfg.parallel_config.engine_worker_queue_port = [7000, 7001]
    engine.cfg.worker_num_per_node = 1
    engine.cfg.node_rank = 0
    engine.cfg.nnode = 1
    engine.cfg.master_ip = "127.0.0.1"
    engine.launched_expert_service_signal = SimpleNamespace(value=np.array([0, 0], dtype=np.int32))

    engine.engine = SimpleNamespace(
        split_connector=SimpleNamespace(start_receiver=lambda: None),
        scheduler=SimpleNamespace(start=lambda *args: None),
    )

    class DummyQueue:
        def __init__(self, *args, **kwargs):
            pass

    class DummyProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            pass

    class DummyContext:
        def Process(self, target, args):
            return DummyProcess(target, args)

    captured = {}

    def fake_queue(**kwargs):
        captured["address"] = kwargs["address"]
        return SimpleNamespace(cleanup=lambda: None)

    def fake_sleep(_):
        engine.launched_expert_service_signal.value[1] = 1

    monkeypatch.setattr(engine_module.multiprocessing, "Queue", DummyQueue)
    monkeypatch.setattr(engine_module.multiprocessing, "get_context", lambda name: DummyContext())
    monkeypatch.setattr(engine_module, "EngineWorkerQueue", fake_queue)
    monkeypatch.setattr(engine_module.envs, "FD_ENABLE_MULTI_API_SERVER", False)
    monkeypatch.setattr(engine_module.envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
    monkeypatch.setattr(engine_module.time, "sleep", fake_sleep)

    engine.launch_components()

    assert captured["address"] == "/dev/shm/fd_task_queue_7001.sock"


def test_check_worker_initialize_status_progress(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.worker_init_status = {}
    engine.worker_ready_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    stdout_lines = [
        b"Loading checkpoint shards: 50",
        b"Start load layer 2",
    ]

    class DummyProc:
        def __init__(self, lines):
            self.stdout = list(lines)

        def poll(self):
            return None

    engine.worker_proc = DummyProc(stdout_lines)

    class DummyPbar:
        def __init__(self, total, desc):
            self.total = total
            self.n = 0

        def update(self, value):
            self.n += value

        def refresh(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", lambda total, desc: DummyPbar(total, desc))
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.check_worker_initialize_status() is True
    assert engine.worker_init_status["finished"] is True


def test_check_worker_initialize_status_early_finish(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.worker_init_status = {"finished": True, "weight_loadding": 1, "layer_loadding": 1}
    engine.worker_ready_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    class DummyProc:
        def __init__(self):
            self.stdout = [b"done"]

        def poll(self):
            return None

    engine.worker_proc = DummyProc()

    class DummyPbar:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, value):
            self.n += value

        def refresh(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", lambda total, desc: DummyPbar(total, desc))
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.check_worker_initialize_status() is True


def test_check_worker_initialize_status_poll_failures(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.worker_init_status = {}
    engine.worker_ready_signal = SimpleNamespace(value=np.array([0], dtype=np.int32))

    class DummyProc:
        def __init__(self):
            self.stdout = []
            self.calls = 0

        def poll(self):
            self.calls += 1
            if self.calls == 1:
                return 1
            return None

    engine.worker_proc = DummyProc()

    class DummyPbar:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, value):
            self.n += value

        def refresh(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", lambda total, desc: DummyPbar(total, desc))
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.check_worker_initialize_status() is False


def test_check_worker_initialize_status_join_exception(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.cfg = build_cfg()
    engine.worker_init_status = {"weight_loadding": 1, "layer_loadding": 1}
    engine.worker_ready_signal = SimpleNamespace(value=np.array([1], dtype=np.int32))

    class DummyProc:
        def __init__(self):
            self.stdout = []

        def poll(self):
            return None

    engine.worker_proc = DummyProc()

    class DummyThread:
        def __init__(self, target, daemon):
            self._target = target

        def start(self):
            pass

        def join(self, timeout=None):
            raise RuntimeError("join failed")

    monkeypatch.setattr(engine_module.threading, "Thread", DummyThread)

    class DummyPbar:
        def __init__(self, total, desc):
            self.n = 0

        def update(self, value):
            self.n += value

        def refresh(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine_module, "tqdm", lambda total, desc: DummyPbar(total, desc))
    monkeypatch.setattr(engine_module.time, "sleep", lambda _: None)

    assert engine.check_worker_initialize_status() is True


def test_exit_sub_services_cleans_resources(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.running = True
    engine.worker_ready_signal = DummySignal("worker", np.zeros([1], dtype=np.int32), np.int32, "s", True)
    engine.loaded_model_signal = DummySignal("loaded", np.zeros([1], dtype=np.int32), np.int32, "s", True)
    engine.get_profile_block_num_signal = DummySignal("profile", np.zeros([1], dtype=np.int32), np.int32, "s", True)

    cache_manager = SimpleNamespace(
        shm_cache_task_flag_broadcast=SimpleNamespace(clear=lambda: None),
        cache_ready_signal=SimpleNamespace(clear=lambda: None),
    )
    engine.engine = SimpleNamespace(resource_manager=SimpleNamespace(cache_manager=cache_manager))
    engine.cache_manager_processes = [SimpleNamespace(pid=10)]
    engine.worker_proc = SimpleNamespace(pid=20)
    engine.zmq_server = SimpleNamespace(close=lambda: None)
    engine.dp_processed = [SimpleNamespace(pid=30, join=lambda: None)]
    engine.dp_engine_worker_queue_server = [SimpleNamespace(cleanup=lambda: None)]

    monkeypatch.setattr(engine_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(engine_module.os, "killpg", lambda pid, sig: None)

    engine._exit_sub_services()

    assert engine.running is False
    assert engine.worker_ready_signal.cleared is True
    assert engine.loaded_model_signal.cleared is True


def test_exit_sub_services_handles_kill_errors(monkeypatch):
    engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
    engine.running = True
    engine.worker_ready_signal = DummySignal("worker", np.zeros([1], dtype=np.int32), np.int32, "s", True)
    engine.loaded_model_signal = DummySignal("loaded", np.zeros([1], dtype=np.int32), np.int32, "s", True)
    engine.cache_manager_processes = [SimpleNamespace(pid=10)]
    engine.engine = SimpleNamespace(resource_manager=SimpleNamespace(cache_manager=SimpleNamespace()))
    engine.worker_proc = SimpleNamespace(pid=20)

    def raise_error(_):
        raise OSError("fail")

    monkeypatch.setattr(engine_module.os, "getpgid", raise_error)
    monkeypatch.setattr(engine_module.os, "killpg", lambda pid, sig: None)

    engine._exit_sub_services()

    assert engine.running is False
