import json
import logging
import time

from llama_cpp import Llama

logger = logging.getLogger("Engine")
logger.setLevel(logging.INFO)


class ChainedEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChainedEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path):
        if not hasattr(self, 'llm'):
            print(f">>> [Engine] Loading Model: {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,  # Mac M1 Metal 加速
                verbose=False
            )
            self._init_prompts()
            print(">>> [Engine] Model Loaded.")
    def _init_prompts(self):
            """定义所有 Prompt 模板"""

            # --- Layer 1: 路由器 (增强版) ---
            # 优化点: 增加了 Few-Shot 示例，明确了"推荐"在不同场景下的归属
            self.router_prompt = """你是一个意图分类器。
    请将用户输入归类为以下三类之一：
    1. RECRUIT (招聘/找工作/查薪资/招人)
    2. ECOMMERCE (电商/买东西/查商品/推荐产品)
    3. CHAT (闲聊/无意义/其他)

    示例:
    "找个java工作" -> RECRUIT
    "苹果手机多少钱" -> ECOMMERCE
    "你好" -> CHAT
    "帮我推荐个好用的鼠标" -> ECOMMERCE
    "推荐几个靠谱的简历" -> RECRUIT
    "200元能买什么" -> ECOMMERCE
    "20k能招到人吗" -> RECRUIT

    只输出类别代码 (RECRUIT, ECOMMERCE, CHAT)，不要解释。
    """

            # --- Layer 2: 垂直专家 (保持不变) ---
            self.expert_prompts = {
                "RECRUIT": """你是一个招聘搜索专家。
    请提取: city(城市), job(职位), salary(薪资), exp(经验)。
    输出 JSON。
    示例: "上海3年经验Java" -> {"city":"上海", "job":"Java", "exp":"3年"}""",

                "ECOMMERCE": """你是一个电商搜索专家。
    请提取: category(品类), brand(品牌), price(价格), color(颜色)。
    输出 JSON。
    示例: "200元左右的红色口红" -> {"category":"口红", "price":"200元", "color":"红色"}"""
            }

    def predict(self, query: str) -> dict:
        start_time = time.time()

        # ===========================
        # Step 1: Layer 1 (Router)
        # ===========================
        router_messages = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": query}
        ]

        # 技巧: max_tokens=5 限制输出长度，强制模型快速结束
        router_out = self.llm.create_chat_completion(
            messages=router_messages,
            temperature=0.1,
            max_tokens=10
        )

        # 清洗结果 (模型可能输出 "类别: RECRUIT"，我们只要 "RECRUIT")
        raw_intent = router_out['choices'][0]['message']['content'].strip()
        intent_category = "CHAT"  # 默认兜底

        if "RECRUIT" in raw_intent.upper():
            intent_category = "RECRUIT"
        elif "ECOMMERCE" in raw_intent.upper():
            intent_category = "ECOMMERCE"

        print(f"   [Layer 1] Router: {intent_category}")

        # ===========================
        # Step 2: Layer 2 (Expert)
        # ===========================
        entities = {}

        # Early Exit: 如果是闲聊，直接跳过 L2，省时间！
        if intent_category == "CHAT":
            pass

        elif intent_category in self.expert_prompts:
            expert_sys_prompt = self.expert_prompts[intent_category]

            expert_messages = [
                {"role": "system", "content": expert_sys_prompt},
                {"role": "user", "content": query}
            ]

            # 技巧: 强制 JSON 模式
            expert_out = self.llm.create_chat_completion(
                messages=expert_messages,
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            try:
                raw_json = expert_out['choices'][0]['message']['content']
                entities = json.loads(raw_json)
                print(f"   [Layer 2] Expert: {entities}")
            except:
                print("   [Layer 2] JSON Parse Failed")

        latency = int((time.time() - start_time) * 1000)

        return {
            "intent": intent_category,
            "entities": entities,
            "latency": latency
        }
