import pandas as pd
import math

# 3. Asignar test al último 20% de cada video
def assign_split(group):
    n_test = math.floor(len(group) * 0.2)
    if n_test > 0:
        group.iloc[-n_test:, group.columns.get_loc('split')] = 'test'
    return group

def load_dataset_with_splits(dataset_path: str, output_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dataset_path)
    
    df['split'] = 'train'
    
    df = df.groupby('video', group_keys=False).apply(assign_split)
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # 5. Formatear columnas
    for split_df in [train_df, test_df]:
        split_df = split_df.rename(columns={
            'id': 'sentence_id',
            'label': 'text'
        })
        if 'clip_id' not in split_df.columns:
            split_df['clip_id'] = split_df['sentence_id']
        split_df['source_split'] = split_df['split']
    
    # 6. Guardar
    train_df.to_csv(f"{output_dir}/train_set.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_set.csv", index=False)
    final_df = pd.concat([train_df,test_df])
    final_df.to_csv("meta_with_splits.csv")
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Total: {len(train_df) + len(test_df)} samples")
    print(f"\nGuardado en {output_dir}/")
    print(f"  - train_set.csv")
    print(f"  - test_set.csv")
    
    return train_df, test_df

# Uso
train_df, test_df = load_dataset_with_splits("meta.csv")