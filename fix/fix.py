import pandas as pd

# 讀取 CSV 文件
file_path = 'submission.csv'
df = pd.read_csv(file_path)

# 檢查和修正 id 欄位
df['id'] = pd.to_numeric(df['id'], errors='coerce')

# 將 NaN 值設為 -1 以便於檢查
df['id'] = df['id'].fillna(-1).astype(int)

# 設定新的連續 id
df['id'] = range(len(df))

# 儲存修正後的 CSV 文件
output_path = 'fix.csv'
df.to_csv(output_path, index=False)

print(f'修正後的文件已儲存在 {output_path}')
