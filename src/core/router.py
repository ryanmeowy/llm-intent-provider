import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("Router")


class SemanticRouter:
    def __init__(self):
        print(">>> [Router] Loading Embedding Model (all-MiniLM-L6-v2)...")
        # 这个模型会自动下载，只有 80MB，非常快
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # === 1. 定义意图库 (模拟通用搜索场景，几十个都没问题) ===
        # 这里的 desc 是给向量模型看的，用来匹配用户的 Query
        self.intent_db = [
            {"code": "RECRUIT", "desc": "招聘, 找工作, 查薪资, 面试, 职位要求, 招聘网站"},
            {"code": "ECOMMERCE", "desc": "买东西, 商品价格, 推荐产品, 购物, 便宜的, 性价比"},
            {"code": "CODING", "desc": "写代码, 编程报错, Python教程, Java异常, 算法逻辑"},
            {"code": "NAVIGATIONAL", "desc": "去哪里, 怎么走, 附近的餐厅, 地图导航, 酒店预订"},
            {"code": "RECIPE", "desc": "怎么做菜, 食谱, 好吃的做法, 烹饪教程"},
            {"code": "MEDICAL", "desc": "感冒了吃什么药, 医院挂号, 身体不舒服, 症状分析"},
            {"code": "CHAT", "desc": "你好, 闲聊, 讲个笑话, 无论什么话题都可以聊"},
        ]

        # === 2. 预计算意图库的向量 (Cold Start) ===
        # 我们把 desc 变成向量存起来，这样推理时只需要算 Query 的向量
        self.intent_vectors = self.embed_model.encode(
            [item["desc"] for item in self.intent_db],
            convert_to_tensor=True
        )
        print(f">>> [Router] Indexed {len(self.intent_db)} intents.")

    def search_top_k(self, query: str, k: int = 3):
        """
        核心逻辑:
        1. 把用户 Query 变成向量
        2. 和意图库算余弦相似度
        3. 返回分数最高的 k 个
        """
        # 1. Encode Query
        query_vec = self.embed_model.encode(query, convert_to_tensor=True)

        # 2. Compute Cosine Similarity
        cos_scores = util.cos_sim(query_vec, self.intent_vectors)[0]

        # 3. Sort & Extract Top K
        # torch.topk 返回 (values, indices)
        top_results = kw_results = torch.topk(cos_scores, k=min(k, len(self.intent_db)))

        candidates = []
        for score, idx in zip(top_results[0], top_results[1]):
            item = self.intent_db[idx]
            candidates.append({
                "code": item["code"],
                "desc": item["desc"],
                "score": score.item()
            })

        return candidates


# 为了让上面代码运行不需要 torch 依赖报错 (sentence-transformers 自带了部分 torch)，
# 这里补一个 import torch，如果没有安装完整 torch，util.cos_sim 依然能工作
import torch