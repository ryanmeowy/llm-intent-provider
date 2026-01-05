import json

import protos.intent_pb2 as pb2
import protos.intent_pb2_grpc as pb2_grpc
from src.core.engine import ChainedEngine


class IntentServiceImpl(pb2_grpc.IntentServiceServicer):

    def __init__(self, model_path):
        # 初始化引擎
        self.engine = ChainedEngine(model_path)

    def Analyze(self, request, context):
        query = request.query

        # 调用核心引擎
        result = self.engine.predict(query)

        # 构造响应
        return pb2.AnalyzeResponse(
            intent_category=result["intent"],
            entities_json=json.dumps(result["entities"], ensure_ascii=False),
            latency_ms=result["latency"]
        )