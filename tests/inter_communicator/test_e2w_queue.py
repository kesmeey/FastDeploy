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

import unittest

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator.engine_worker_queue import EngineWorkerQueue
from fastdeploy.utils import to_numpy, to_tensor


class DummyTask:
    def __init__(self, images):
        self.multimodal_inputs = {"images": images}


class TestEngineWorkerQueue(unittest.TestCase):
    def _create_queue_pair(self, address=("127.0.0.1", 0), num_client=1):
        server = EngineWorkerQueue(
            address=address,
            authkey=b"secret_key",
            is_server=True,
            num_client=num_client,
        )
        client = EngineWorkerQueue(
            address=server.address,
            authkey=b"secret_key",
            is_server=False,
            num_client=num_client,
            client_id=0,
        )
        return server, client

    def test_to_tensor_success(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)
        tasks = [task]
        to_tensor(tasks)

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_tensor_disabled(self):
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)
        tasks = [task]
        to_tensor(tasks)

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_tensor_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()
        tasks = [task]

        # 不应抛异常
        to_tensor(tasks)
        self.assertFalse(hasattr(task, "multimodal_inputs"))

    def test_to_tensor_exception_handling(self):
        bad_task = DummyTask(images="not an array")
        bad_tasks = [bad_task]

        to_tensor(bad_tasks)
        self.assertEqual(bad_task.multimodal_inputs["images"], "not an array")

    def test_to_numpy_success(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        # 构造 paddle.Tensor 输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        task = DummyTask(tensor_images)
        tasks = [task]
        to_numpy(tasks)

        # 验证转换为 numpy.ndarray
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_disabled(self):
        # 创建随机张量作为测试输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        # 创建模拟任务
        task = DummyTask(tensor_images)
        tasks = [task]

        # 调用转换方法(预期不会转换)
        to_numpy(tasks)

        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()
        tasks = [task]

        # 不应抛异常
        to_numpy(tasks)
        self.assertFalse(hasattr(task, "multimodal_inputs"))

    def test_to_numpy_non_tensor_input(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        np_images = np.random.randn(2, 3, 224, 224)
        task = DummyTask(np_images)
        tasks = [task]

        to_numpy(tasks)

        # 非 Tensor 输入应保持为 numpy 数组
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_exception_handling(self):
        envs.FD_ENABLE_MAX_PREFILL = 1

        # 构造错误输入（让 .numpy() 抛异常）
        class BadTensor:
            def numpy(self):
                raise RuntimeError("mock error")

        bad_task = DummyTask(images=None)
        bad_task.multimodal_inputs["image_features"] = [BadTensor()]
        bad_tasks = [bad_task]

        to_numpy(bad_tasks)
        self.assertIsInstance(bad_task.multimodal_inputs["image_features"][0], BadTensor)

    def test_features_info_to_tensor(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        np_feature = paddle.randn([2, 3, 224, 224]).numpy()
        multimodal_inputs = {
            "image_features": [np_feature, np_feature],
        }
        req_dict = {
            "request_id": "req1",
            "multimodal_inputs": multimodal_inputs,
        }
        task = Request.from_dict(req_dict)
        to_tensor([task])

        # 验证已转换为tensor
        self.assertEqual(len(task.multimodal_inputs["image_features"]), 2)
        self.assertIsInstance(task.multimodal_inputs["image_features"][0], paddle.Tensor)
        self.assertIsInstance(task.multimodal_inputs["image_features"][1], paddle.Tensor)

    def test_features_info_to_numpy(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        tensor_feature = paddle.randn([2, 3, 224, 224])
        multimodal_inputs = {
            "video_features": [tensor_feature, tensor_feature],
        }
        req_dict = {
            "request_id": "req1",
            "multimodal_inputs": multimodal_inputs,
        }
        task = Request.from_dict(req_dict)
        to_numpy([task])

        # 验证已转换为ndarray
        self.assertEqual(len(task.multimodal_inputs["video_features"]), 2)
        self.assertIsInstance(task.multimodal_inputs["video_features"][0], np.ndarray)
        self.assertIsInstance(task.multimodal_inputs["video_features"][1], np.ndarray)

    def test_engine_worker_queue_client_signals_and_port(self):
        server, client = self._create_queue_pair()
        try:
            self.assertIsNone(client.exist_tasks_intra_signal)
            self.assertFalse(client.exist_tasks())
            client.set_exist_tasks(True)
            self.assertTrue(client.exist_tasks())
            client.set_exist_tasks(False)
            self.assertFalse(client.exist_tasks())
            with self.assertRaises(RuntimeError):
                client.get_server_port()
        finally:
            server.cleanup()

    def test_engine_worker_queue_connect_retry_failure(self):
        class RefusingManager:
            def connect(self):
                raise ConnectionRefusedError("refuse")

        queue = EngineWorkerQueue.__new__(EngineWorkerQueue)
        queue.manager = RefusingManager()
        queue.address = ("127.0.0.1", 12345)
        with self.assertRaises(ConnectionError):
            queue._connect_with_retry(max_retries=2, interval=0)

    def test_engine_worker_queue_task_flow(self):
        server, client = self._create_queue_pair()
        original_flag = envs.FD_ENABLE_E2W_TENSOR_CONVERT
        envs.FD_ENABLE_E2W_TENSOR_CONVERT = 0
        try:
            np_images = paddle.randn([1, 3, 8, 8]).numpy()
            task = DummyTask(np_images)
            tasks = [[task]]
            client.put_tasks(tasks)

            self.assertTrue(client.exist_tasks())
            self.assertEqual(client.num_tasks(), 1)

            got_tasks, all_read = client.get_tasks()
            self.assertTrue(all_read)
            self.assertFalse(client.exist_tasks())
            self.assertIsInstance(got_tasks[0][0][0].multimodal_inputs["images"], np.ndarray)
            self.assertEqual(client.num_tasks(), 0)

            client.clear_data()
            self.assertEqual(client.num_tasks(), 0)
            self.assertEqual(list(client.client_read_flag), [1])
        finally:
            envs.FD_ENABLE_E2W_TENSOR_CONVERT = original_flag
            server.cleanup()

    def test_engine_worker_queue_connect_cache_and_disaggregate(self):
        server, client = self._create_queue_pair()
        try:
            empty_task, all_read = client.get_connect_rdma_task()
            self.assertIsNone(empty_task)
            self.assertTrue(all_read)

            client.put_connect_rdma_task({"id": 7})
            connect_task, all_read = client.get_connect_rdma_task()
            self.assertTrue(all_read)
            self.assertEqual(connect_task["id"], 7)

            self.assertIsNone(client.get_connect_rdma_task_response())
            self.assertTrue(client.put_connect_rdma_task_response({"success": True}))
            task_response = client.get_connect_rdma_task_response()
            self.assertEqual(task_response, {"success": True})

            self.assertEqual(client.get_cache_info(), [])
            client.put_cache_info([{"cache": "v1"}])
            self.assertEqual(client.num_cache_infos(), 1)
            cache_infos = client.get_cache_info()
            self.assertEqual(cache_infos, [{"cache": "v1"}])
            self.assertEqual(client.get_cache_info(), [])
            self.assertEqual(client.num_cache_infos(), 0)

            self.assertTrue(client.put_finished_req([["req1", {"status": "ok"}]]))
            client.finished_send_cache_list.append(["req1", {"error": "fail"}])
            finished = client.get_finished_req()
            self.assertEqual(finished, [["req1", {"error": "fail"}]])
            self.assertEqual(client.get_finished_req(), [])

            self.assertTrue(client.put_finished_add_cache_task_req(["req-a"]))
            add_cache_finished = client.get_finished_add_cache_task_req()
            self.assertEqual(add_cache_finished, ["req-a"])

            self.assertTrue(client.disaggregate_queue_empty())
            self.assertIsNone(client.get_disaggregated_tasks())
            client.put_disaggregated_tasks({"task": 1})
            client.put_disaggregated_tasks({"task": 2})
            items = client.get_disaggregated_tasks()
            self.assertEqual(items, [{"task": 1}, {"task": 2}])
            self.assertTrue(client.disaggregate_queue_empty())
        finally:
            server.cleanup()

    def test_engine_worker_queue_put_tasks_tensor_conversion(self):
        """Test put_tasks with tensor conversion enabled"""
        server, client = self._create_queue_pair()
        original_flag = envs.FD_ENABLE_E2W_TENSOR_CONVERT
        envs.FD_ENABLE_E2W_TENSOR_CONVERT = 1
        try:
            # Create task with numpy array that should be converted to tensor
            np_images = paddle.randn([1, 3, 8, 8]).numpy()
            task = DummyTask(np_images)
            tasks = [[task]]

            # Manually set client_read_flag to simulate other clients not ready
            client.client_read_flag[0] = 0  # Not ready yet

            client.put_tasks(tasks)

            # Verify tensor conversion happened
            self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)
        finally:
            envs.FD_ENABLE_E2W_TENSOR_CONVERT = original_flag
            server.cleanup()

    def test_engine_worker_queue_put_connect_rdma_task_wait(self):
        """Test put_connect_rdma_task waits for all clients"""
        server, client = self._create_queue_pair(num_client=2)
        try:
            # Create second client
            client2 = EngineWorkerQueue(
                address=server.address,
                authkey=b"secret_key",
                is_server=False,
                num_client=2,
                client_id=1,
            )

            # Set client flags to simulate waiting
            client.client_get_connect_task_flag[1] = 0  # Second client not ready

            # This should wait and then succeed
            client.put_connect_rdma_task({"test": "data"})

            # Verify task was set
            self.assertEqual(len(server.connect_rdma_tasks), 1)
            self.assertEqual(server.connect_rdma_tasks[0], {"test": "data"})

            client2.cleanup()
        finally:
            server.cleanup()

    def test_engine_worker_queue_put_connect_response_wait(self):
        """Test put_connect_rdma_task_response waits for flag"""
        server, client = self._create_queue_pair(num_client=2)
        try:
            # Set up initial state
            client.can_put_next_connect_task_response_flag.set(0)  # Not ready

            # This should wait for the flag
            import threading
            import time

            def set_flag():
                time.sleep(0.01)  # Small delay
                client.can_put_next_connect_task_response_flag.set(1)

            thread = threading.Thread(target=set_flag)
            thread.start()

            result = client.put_connect_rdma_task_response({"success": True})

            thread.join()
            # Should return False since not all clients have responded yet
            self.assertFalse(result)

        finally:
            server.cleanup()

    def test_engine_worker_queue_get_connect_response_wait(self):
        """Test get_connect_rdma_task_response waits for all clients"""
        server, client = self._create_queue_pair(num_client=2)
        try:
            # Create second client and set responses
            client2 = EngineWorkerQueue(
                address=server.address,
                authkey=b"secret_key",
                is_server=False,
                num_client=2,
                client_id=1,
            )

            # Both clients put responses
            client.put_connect_rdma_task_response({"success": True})
            client2.put_connect_rdma_task_response({"success": False})

            # Now get should work
            response = client.get_connect_rdma_task_response()
            self.assertEqual(response["success"], False)  # Should be AND of all responses

            client2.cleanup()
        finally:
            server.cleanup()

    def test_engine_worker_queue_put_cache_info_wait(self):
        """Test put_cache_info waits for all clients"""
        server, client = self._create_queue_pair(num_client=2)
        try:
            client2 = EngineWorkerQueue(
                address=server.address,
                authkey=b"secret_key",
                is_server=False,
                num_client=2,
                client_id=1,
            )

            # Set client flags to simulate waiting
            client.client_read_info_flag[1] = 0  # Second client not ready

            cache_info = [{"cache_id": "test"}]
            client.put_cache_info(cache_info)

            # Verify cache info was set
            self.assertEqual(server.cache_infos, cache_info)

            client2.cleanup()
        finally:
            server.cleanup()

    def test_engine_worker_queue_put_finished_req_wait(self):
        """Test put_finished_req waits for flag"""
        server, client = self._create_queue_pair()
        try:
            # Set flag to not ready
            client.can_put_next_send_cache_finished_flag.set(0)

            # This should wait
            def set_flag():
                import time

                time.sleep(0.01)
                client.can_put_next_send_cache_finished_flag.set(1)

            import threading

            thread = threading.Thread(target=set_flag)
            thread.start()

            result = client.put_finished_req([["req1", {"status": "ok"}]])

            thread.join()
            self.assertTrue(result)

        finally:
            server.cleanup()

    def test_engine_worker_queue_get_finished_req_wait(self):
        """Test get_finished_req waits for all clients"""
        server, client = self._create_queue_pair(num_client=2)
        try:
            client2 = EngineWorkerQueue(
                address=server.address,
                authkey=b"secret_key",
                is_server=False,
                num_client=2,
                client_id=1,
            )

            # Both clients put finished requests
            client.put_finished_req([["req1", {"status": "ok"}]])
            client2.put_finished_req([["req1", {"error": "fail"}]])

            # Get should return the one with error
            result = client.get_finished_req()
            self.assertEqual(result[0][1]["error"], "fail")

            client2.cleanup()
        finally:
            server.cleanup()

    def test_engine_worker_queue_put_add_cache_task_wait(self):
        """Test put_finished_add_cache_task_req waits for flag"""
        server, client = self._create_queue_pair()
        try:
            # Set flag to not ready
            client.can_put_next_add_task_finished_flag.set(0)

            def set_flag():
                import time

                time.sleep(0.01)
                client.can_put_next_add_task_finished_flag.set(1)

            import threading

            thread = threading.Thread(target=set_flag)
            thread.start()

            result = client.put_finished_add_cache_task_req(["req1"])

            thread.join()
            self.assertTrue(result)

        finally:
            server.cleanup()

    def test_engine_worker_queue_get_add_cache_task_wait(self):
        """Test get_finished_add_cache_task_req waits for all clients"""
        server, client = self._create_queue_pair(num_client=2)
        try:
            client2 = EngineWorkerQueue(
                address=server.address,
                authkey=b"secret_key",
                is_server=False,
                num_client=2,
                client_id=1,
            )

            # Both clients put finished add cache tasks
            client.put_finished_add_cache_task_req(["req1"])
            client2.put_finished_add_cache_task_req(["req2"])

            # Get should work (returns first one)
            result = client.get_finished_add_cache_task_req()
            self.assertIn(result, [["req1"], ["req2"]])

            client2.cleanup()
        finally:
            server.cleanup()


if __name__ == "__main__":
    unittest.main()
