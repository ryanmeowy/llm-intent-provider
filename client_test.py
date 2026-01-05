import os
import sys

import grpc

sys.path.append(os.path.abspath("protos"))
import protos.intent_pb2 as pb2
import protos.intent_pb2_grpc as pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.IntentServiceStub(channel)

    test_cases = [
        "你好啊",                          # 期望: CHAT (Fast)
        "帮我找个上海的算法工程师",          # 期望: RECRUIT (Slow, Structured)
        "推荐个200块钱的机械键盘",          # 期望: ECOMMERCE (Slow, Structured)
        "感冒了怎么办"
    ]

    for q in test_cases:
        print(f"\n>>> 发送: {q}")
        try:
            resp = stub.Analyze(pb2.AnalyzeRequest(query=q))
            print(f"   [意图]: {resp.intent_category}")
            print(f"   [实体]: {resp.entities_json}")
            print(f"   [耗时]: {resp.latency_ms} ms")
        except grpc.RpcError as e:
            print(f"RPC Error: {e}")

if __name__ == '__main__':
    run()