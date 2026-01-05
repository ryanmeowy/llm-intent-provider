import os
import sys
import grpc
from concurrent import futures

# 添加路径，确保能找到 protos
sys.path.append(os.path.abspath("protos"))

import protos.intent_pb2_grpc as pb2_grpc
from src.service.grpc_server import IntentServiceImpl


def serve():
    # 配置
    MODEL_PATH = "./qwen2.5-1.5b-instruct-q4_k_m.gguf"
    PORT = "50051"

    # 启动服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_IntentServiceServicer_to_server(
        IntentServiceImpl(MODEL_PATH), server
    )

    server.add_insecure_port(f'[::]:{PORT}')
    print(f">>> Server started on port {PORT}")
    print(f">>> Ready to accept requests...")

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()