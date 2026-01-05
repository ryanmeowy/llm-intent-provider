import time
import json
import logging
from llama_cpp import Llama

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Engine")


class ChainedEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChainedEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path):
        if not hasattr(self, 'llm'):
            print(f">>> [Init] Loading LLM: {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # 加大上下文窗口，确保能装下长 Prompt
                n_gpu_layers=-1,  # 开启 Metal 加速
                verbose=True  # 开启日志，方便看底层有没有报错
            )
            self._init_prompts()
            print(">>> [Init] Engine Loaded.")

    def _init_prompts(self):
        # ==========================================
        # Layer 1: 强力路由 Prompt (All-in-One)
        # 核心技巧：用【示例】代替【定义】，1.5B 模型模仿能力强，理解能力弱
        # ==========================================
        self.router_prompt = """你是一个智能搜索路由。
        请分析用户输入，将其归类为以下核心领域之一。
        如果无法归类到特定领域，或者只是查询普通信息，请归类为 WEB_SEARCH。

        核心领域 (需要结构化):
        1. ECOMMERCE: (电商) 买东西, 查价格, 品牌, 产品推荐
        2. LOCAL: (本地生活) 找餐厅, 酒店, 景点, 导航, 附近
        3. MEDIA: (影音书) 找电影, 听歌, 找小说, 电视剧
        4. CHAT: (闲聊) 打招呼, 情感交流, 无意义语句

        通用兜底 (不需要结构化):
        5. WEB_SEARCH: (通用搜索) 新闻, 历史, 百科, 教程, 怎么做(非菜谱), 为什么, 是什么, 以及所有不属于上述类别的内容

        示例:
        "买个好用的吹风机" -> ECOMMERCE
        "附近的火锅店" -> LOCAL
        "周杰伦的歌" -> MEDIA
        "你好" -> CHAT
        "感冒了怎么办" -> WEB_SEARCH  (医疗建议归为通用搜索)
        "特朗普最新新闻" -> WEB_SEARCH (新闻归为通用搜索)
        "相对论是谁提出的" -> WEB_SEARCH (百科归为通用搜索)
        "红烧肉怎么做" -> WEB_SEARCH (菜谱如果没做垂直域，就归为通用)

        只输出分类代码，不要解释。
        User: "{query}" -> """

        # ==========================================
        # Layer 2: 垂直专家 Prompt
        # ==========================================
        self.expert_prompts = {
            "RECRUIT": "你是一个招聘专家。提取: city, job, salary, exp。如果用户未提及某字段，请填空字符串。输出JSON。示例: '上海3年Java' -> {\"city\":\"上海\", \"job\":\"Java\", \"exp\":\"3年\"}",
            "ECOMMERCE": "你是一个电商专家。提取: category, brand, price, color。如果用户未提及某字段，请填空字符串。输出JSON。示例: '200元红色口红' -> {\"category\":\"口红\", \"price\":\"200元\", \"color\":\"红色\"}",
            "CODING": "你是一个编程助手。提取: language, framework, error。如果用户未提及某字段，请填空字符串。输出JSON。",
            "RECIPE": "你是一个大厨。提取: dish(菜名), ingredient(食材)。如果用户未提及某字段，请填空字符串。输出JSON。"
        }

    def predict(self, query: str) -> dict:
        start_time = time.time()

        # --- Step 1: Layer 1 Router ---
        # 直接利用 f-string 构造完整 Prompt
        final_router_prompt = self.router_prompt.replace("{query}", query)

        # 技巧: max_tokens=5, stop=["\n"], temperature=0
        # 极致限制模型的发挥空间，让它只能吐出代码
        router_out = self.llm(
            final_router_prompt,
            max_tokens=10,
            stop=["\n"],
            temperature=0.0,  # 绝对理性，不要随机
            echo=False
        )

        raw_intent = router_out['choices'][0]['text'].strip().upper()

        # 清洗结果: 只要是咱们定义的词，包含就算命中
        final_intent = "CHAT"
        valid_intents = ["RECRUIT", "ECOMMERCE", "CODING", "RECIPE", "CHAT"]

        for valid in valid_intents:
            if valid in raw_intent:
                final_intent = valid
                break

        print(f"   [Layer 1] Input: {query} | Output: {raw_intent} | Decision: {final_intent}")

        # --- Step 2: Layer 2 Expert ---
        entities = {}

        # Early Exit
        if final_intent == "CHAT":
            pass
        elif final_intent in self.expert_prompts:
            sys_prompt = self.expert_prompts[final_intent]
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ]

            expert_out = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            try:
                entities = json.loads(expert_out['choices'][0]['message']['content'])
                print(f"   [Layer 2] Extracted: {entities}")
            except:
                print("   [Layer 2] JSON Parse Error")

        latency = int((time.time() - start_time) * 1000)

        return {
            "intent": final_intent,
            "entities": entities,
            "latency": latency
        }
