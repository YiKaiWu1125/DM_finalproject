import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 讀取數據
train_behaviors = pd.read_csv('train/train_behaviors.tsv', sep='\t', names=['id', 'user_id', 'time', 'history', 'impressions'], dtype={'id': str, 'user_id': str, 'time': str, 'history': str, 'impressions': str})
train_news = pd.read_csv('train/train_news.tsv', sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
entity_embedding = pd.read_csv('train/train_entity_embedding.vec', sep='\t', header=None)

# 檢查數據
print("Sample of train_behaviors:")
print(train_behaviors.head())
print("Sample of train_news:")
print(train_news.head())

# 合併新聞標題和摘要進行 BERT 編碼
train_news['content'] = train_news['title'].fillna('') + ' ' + train_news['abstract'].fillna('')

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
news_dataset = NewsDataset(train_news['content'])
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
train_news['embedding'] = list(news_embeddings)
news_dict = train_news.set_index('news_id')['embedding'].to_dict()

# 處理行為數據
def process_behavior(row):
    history = row['history'].split() if row['history'] else []
    impressions = [imp.split('-') for imp in row['impressions'].split() if '-' in imp]
    print(f"Processing row id {row['id']}: history={history}, impressions={impressions}")
    return history, impressions

train_behaviors['history'], train_behaviors['impressions'] = zip(*train_behaviors.apply(process_behavior, axis=1))

# 檢查數據處理結果
print("Processed train_behaviors sample:")
print(train_behaviors.head())

# 訓練和測試集劃分
train_data, val_data = train_test_split(train_behaviors, test_size=0.2, random_state=42)

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

# 訓練函數
def train_model(train_data, val_data, model, optimizer, criterion, epochs=10, save_path='model.pth'):
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc=f"Training Epoch {epoch+1}/{epochs}"):
            user_history = [news_dict[news_id] for news_id in row['history'] if news_id in news_dict]
            if not user_history:
                continue
            user_embedding = np.mean(user_history, axis=0)
            user_embedding = torch.tensor(user_embedding, dtype=torch.float32).to(device)
            
            for news_id, label in row['impressions']:
                if news_id not in news_dict:
                    continue
                news_embedding = torch.tensor(news_dict[news_id], dtype=torch.float32).to(device)
                label = torch.tensor(float(label), dtype=torch.float32).to(device)
                
                optimizer.zero_grad()
                prediction = model(user_embedding, news_embedding)
                loss = criterion(prediction.squeeze(), label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        val_auc = evaluate_model(val_data, model)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}, Val AUC: {val_auc}")

        # 儲存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with AUC: {val_auc}")

# 評估函數
def evaluate_model(val_data, model):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Evaluating Model"):
            user_history = [news_dict[news_id] for news_id in row['history'] if news_id in news_dict]
            if not user_history:
                continue
            user_embedding = np.mean(user_history, axis=0)
            user_embedding = torch.tensor(user_embedding, dtype=torch.float32).to(device)
            
            for news_id, label in row['impressions']:
                if news_id not in news_dict:
                    continue
                news_embedding = torch.tensor(news_dict[news_id], dtype=torch.float32).to(device)
                label = float(label)
                
                prediction = model(user_embedding, news_embedding).cpu().numpy().squeeze()
                predictions.append(prediction)
                labels.append(label)
    
    auc = roc_auc_score(labels, predictions)
    return auc

# 模型訓練
input_dim = 768  # BERT hidden size
hidden_dim = 128
model = RecommenderModel(input_dim * 2, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

train_model(train_data, val_data, model, optimizer, criterion)
