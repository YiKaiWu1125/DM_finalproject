import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 讀取數據
test_behaviors = pd.read_csv('test/test_behaviors.tsv', sep='\t', names=['id', 'user_id', 'time', 'history', 'impressions'], dtype={'id': str, 'user_id': str, 'time': str, 'history': str, 'impressions': str})
test_news = pd.read_csv('test/test_news.tsv', sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

# 檢查數據
print("Sample of test_behaviors:")
print(test_behaviors.head())
print("Sample of test_news:")
print(test_news.head())

# 合併新聞標題和摘要進行 BERT 編碼
test_news['content'] = test_news['title'].fillna('') + ' ' + test_news['abstract'].fillna('')

# BERT Tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 定義 Dataset
class NewsDataset(Dataset):
    def __init__(self, news):
        self.news = news
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.news)
    
    def __getitem__(self, idx):
        content = self.news.iloc[idx]
        inputs = self.tokenizer(content, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        return inputs

# 創建 DataLoader
news_dataset = NewsDataset(test_news['content'])
news_loader = DataLoader(news_dataset, batch_size=16, shuffle=False)

# 獲取 BERT 嵌入
news_embeddings = []
bert_model.eval()
with torch.no_grad():
    for batch in tqdm(news_loader, desc="Encoding news with BERT"):
        inputs = {key: val.squeeze().to(device) for key, val in batch.items()}
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        news_embeddings.append(embeddings.cpu().numpy())

news_embeddings = np.vstack(news_embeddings)

# 將嵌入添加到新聞數據集中
test_news['embedding'] = list(news_embeddings)
news_dict = test_news.set_index('news_id')['embedding'].to_dict()

# 處理行為數據
def process_behavior(row):
    history = row['history'].split() if row['history'] else []
    return history

test_behaviors['history'] = test_behaviors.apply(process_behavior, axis=1)

# 檢查數據處理結果
print("Processed test_behaviors sample:")
print(test_behaviors.head())

# 定義模型
class RecommenderModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RecommenderModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, user_embedding, news_embedding):
        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)
        x = torch.cat([user_embedding, news_embedding], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 加載最佳模型權重
input_dim = 768  # BERT hidden size
hidden_dim = 128
model = RecommenderModel(input_dim * 2, hidden_dim).to(device)
model.load_state_dict(torch.load('model.pth'))
print("Model loaded successfully.")

# 生成提交檔案
def generate_submission(test_data, model, news_dict, output_file='submission.csv'):
    model.eval()
    results = []
    news_ids = list(news_dict.keys())
    with torch.no_grad():
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Generating Submission"):
            user_history = [news_dict[news_id] for news_id in row['history'] if news_id in news_dict]
            if not user_history:
                user_embedding = np.zeros((input_dim,))
            else:
                user_embedding = np.mean(user_history, axis=0)
            user_embedding = torch.tensor(user_embedding, dtype=torch.float32).to(device)
            
            row_result = []
            for _ in range(15):
                news_id = np.random.choice(news_ids)
                news_embedding = torch.tensor(news_dict.get(news_id, np.zeros(input_dim)), dtype=torch.float32).to(device)
                prediction = model(user_embedding, news_embedding).cpu().numpy().squeeze()
                row_result.append(prediction)
            results.append(row_result + [row['id']])  # 確保 id 在最後

    submission_df = pd.DataFrame(results, columns=[f'p{i+1}' for i in range(15)] + ['id'])
    submission_df = submission_df[submission_df['id'] != 'id']  # 移除標題行
    submission_df['id'] = submission_df['id'].astype(int)  # 確保 id 列是整數
    submission_df.to_csv(output_file, index=False)

# 生成提交檔案
generate_submission(test_behaviors, model, news_dict)
print("Submission file generated successfully.")
