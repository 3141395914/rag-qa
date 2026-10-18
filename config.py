"""RAG 问答的配置。"""

# 本地向量模型。第一次运行会从 HuggingFace 下载，
# 国内网络可以先设置环境变量：HF_ENDPOINT=https://hf-mirror.com
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 模型检索时建议给问题加这句前缀，效果会更好
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 切块参数：每块最多 300 字，相邻块重叠 60 字
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

# 检索时取相似度最高的几段
TOP_K = 4

# DeepSeek 接口
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TIMEOUT = 60

# 索引存放目录
INDEX_DIR = "index"
