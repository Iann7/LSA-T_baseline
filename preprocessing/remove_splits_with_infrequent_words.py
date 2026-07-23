import pandas as pd
from collections import Counter
# 1. Cargar datos
df = pd.read_csv("meta_with_splits.csv")
train = df[df['split'] == 'train']
test = df[df['split'] == 'test']

# 2. Contar palabras en TODO (para identificar infrecuentes)
word_counts = Counter()
for label in df['label']:  # Todo el dataset
    word_counts.update(str(label).split())

# 3. Identificar infrecuentes
threshold = 2
infrequent = {w for w, c in word_counts.items() if c < threshold}

# 4. Filtrar SOLO el train
def has_infrequent(label):
    return any(w in infrequent for w in str(label).split())

train_filtered = train[~train['label'].apply(has_infrequent)]

# 5. EL TEST QUEDA IGUAL (NO se filtra)
# test_filtered = test  # ¡NO!

# 6. Combinar
final_df = pd.concat([train_filtered, test], ignore_index=True)
print(len(train_filtered))
print(len(test))