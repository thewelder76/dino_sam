python3 -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models/bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased', cache_dir='./models/bert-base-uncased')
"

