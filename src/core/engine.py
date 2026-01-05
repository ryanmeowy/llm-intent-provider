import time
import json
import logging
import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

# 配置日志
logger = logging.getLogger("Engine")
logger.setLevel(logging.INFO)


class ChainedEngine:
    """
    核心引擎类 (单例模式)
    包含:
    1. 向量模型 (all-MiniLM-L6-v2) -> 用于 Layer 1 粗排
    2. 大语言模型 (Qwen2.5-1.5B)   -> 用于 Layer 1 决策 + Layer 2 提取
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChainedEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path):
        # 防止重复初始化
        if not hasattr(self, 'llm'):
            print(">>> [Init] 正在初始化引擎，请稍候...")

            # --- 1. 加载 LLM (Qwen) ---
            print(f"   - Loading LLM: {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,  # Mac Metal 加速 / Linux CPU
                verbose=False
            )

            # --- 2. 加载向量模型 (用于意图检索) ---
            print("   - Loading Embedding Model (all-MiniLM-L6-v2)...")
            # 首次运行会自动下载约 80MB 的模型文件
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

            # --- 3. 初始化意图库 (定义通用场景) ---
            self._init_intent_database()

            # --- 4. 初始化专家 Prompt ---
            self._init_expert_prompts()

            print(">>> [Init] 引擎初始化完成!")

    def _init_intent_database(self):
        """
        定义 Layer 1 的意图库。
        'code': 意图代码
        'desc': 给向量模型看的描述，用来匹配用户 Query
        """
        self.intent_db = [
            {"code": "RECRUIT", "desc": "招聘, 找工作, 查薪资, 面试, 职位要求, 招聘网站, 招人, 简历"},
            {"code": "ECOMMERCE", "desc": "买东西, 商品价格, 推荐产品, 购物, 便宜的, 性价比, 多少钱, 哪里买"},
            {"code": "CODING", "desc": "写代码, 编程报错, Python教程, Java异常, 算法逻辑, 开发文档, 数据库"},
            {"code": "NAVIGATIONAL", "desc": "去哪里, 怎么走, 附近的餐厅, 地图导航, 酒店预订, 旅游攻略"},
            {"code": "RECIPE", "desc": "怎么做菜, 食谱, 好吃的做法, 烹饪教程, 烘焙, 食材处理"},
            {"code": "MEDICAL", "desc": "感冒了吃什么药, 医院挂号, 身体不舒服, 症状分析, 养生, 医生"},
            {"code": "CHAT", "desc": "你好, 闲聊, 讲个笑话, 无论什么话题都可以聊, 打招呼"},
        ]

        # 预计算意图库的向量 (Cache)
        # 类似于 Java 的 List<Float[]>
        print("   - Indexing Intent Database...")
        descriptions = [item["desc"] for item in self.intent_db]
        self.db_vectors = self.embed_model.encode(descriptions, convert_to_tensor=True)

    def _init_expert_prompts(self):
        """定义 Layer 2 的垂直领域提取规则"""
        self.expert_prompts = {
            "RECRUIT": "你是一个招聘搜索专家。请提取: city(城市), job(职位), salary(薪资), exp(经验)。输出 JSON。",
            "ECOMMERCE": "你是一个电商搜索专家。请提取: category(品类), brand(品牌), price(价格), color(颜色)。输出 JSON。",
            "CODING": "你是一个编程助手。请提取: language(语言), framework(框架), error_msg(错误信息)。输出 JSON。",
            "RECIPE": "你是一个大厨。请提取: dish(菜名), ingredient(食材)。输出 JSON。",
            "MEDICAL": "你是一个医生。请提取: symptom(症状), medicine(药品)。输出 JSON。"
        }

    def _get_top_k_intents(self, query: str, k=3):
        """
        Layer 1 核心逻辑: 向量检索
        输入 Query -> 找出最相似的 k 个意图
        """
        # 1. 把用户 Query 转成向量
        query_vec = self.embed_model.encode(query, convert_to_tensor=True)

        # 2. 计算相似度 (Cosine Similarity)
        cos_scores = util.cos_sim(query_vec, self.db_vectors)[0]

        # 3. 取前 k 名
        top_results = torch.topk(cos_scores, k=min(k, len(self.intent_db)))

        candidates = []
        for score, idx in zip(top_results[0], top_results[1]):
            item = self.intent_db[idx]
            candidates.append({
                "code": item["code"],
                "desc": item["desc"]
            })
        return candidates

    def predict(self, query: str) -> dict:
        """主入口: 执行两层识别"""
        start_time = time.time()

        # ==========================================
        # Step 1: Layer 1 (动态路由)
        # ==========================================

        # 1.1 先用小模型(向量)筛选出 3 个候选者
        candidates = self._get_top_k_intents(query, k=3)
        candidate_codes = [c['code'] for c in candidates]
        print(f"   [Layer 1] Vector Search Candidates: {candidate_codes}")

        # 1.2 动态组装 Prompt
        options_text = "\n".join([f"- {c['code']}: {c['desc']}" for c in candidates])

        router_prompt = f"""你是一个意图分类器。
请从以下候选列表中，选择最匹配用户输入的一个意图代码。

候选列表:
{options_text}

输入: "{query}"
只输出意图代码，不要解释。
"""

        # 1.3 LLM 做最终裁决
        router_msg = [{"role": "user", "content": router_prompt}]
        router_out = self.llm.create_chat_completion(
            messages=router_msg, temperature=0.1, max_tokens=10
        )

        raw_intent = router_out['choices'][0]['message']['content'].strip().upper()

        # 简单清洗: 确保 LLM 输出的是我们候选列表里的词
        final_intent = "CHAT"  # 兜底
        for code in candidate_codes:
            if code in raw_intent:
                final_intent = code
                break

        print(f"   [Layer 1] LLM Decision: {final_intent}")

        # ==========================================
        # Step 2: Layer 2 (专家提取)
        # ==========================================
        entities = {}

        # Early Exit: 闲聊或导航类通常不需要复杂提取，直接跳过以节省时间
        skip_layer_2 = final_intent in ["CHAT", "NAVIGATIONAL"]

        if not skip_layer_2 and final_intent in self.expert_prompts:
            sys_prompt = self.expert_prompts[final_intent]

            expert_msg = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ]

            # 强制 JSON 模式
            expert_out = self.llm.create_chat_completion(
                messages=expert_msg,
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            try:
                json_str = expert_out['choices'][0]['message']['content']
                entities = json.loads(json_str)
                print(f"   [Layer 2] Extracted: {entities}")
            except:
                print("   [Layer 2] JSON Parse Failed")

        latency = int((time.time() - start_time) * 1000)

        return {
            "intent": final_intent,
            "entities": entities,
            "latency": latency
        }