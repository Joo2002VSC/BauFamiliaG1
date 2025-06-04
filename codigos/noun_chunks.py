import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Carregar dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_entrevistas.csv")

# Juntar todo o conteúdo textual
texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar com spaCy
doc = nlp(texto)

def clean_chunk(text):
    text = text.strip(",.\"'")
    # Remover vários tipos de traços no início
    while text.startswith(("-", "–", "—")):
        text = text[1:].strip()
    return text.lower()

# Extrair noun chunks com 2+ palavras, limpos
chunks = [
    clean_chunk(chunk.text)
    for chunk in doc.noun_chunks
    if len(clean_chunk(chunk.text).split()) >= 2
]

# Contar frequência
contagem = Counter(chunks)

# Converter para DataFrame
df_chunks = pd.DataFrame(contagem.items(), columns=["Noun Chunk", "Frequência"])
df_chunks = df_chunks.sort_values(by="Frequência", ascending=False)
df_chunks.to_csv("noun_chunks_frequencia.csv", index=False)

# Gráfico Top 10
top_10 = df_chunks.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10["Noun Chunk"], top_10["Frequência"], color="orange")
plt.title("Top 10 Noun Chunks mais Frequentes")
plt.xlabel("Noun Chunk")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("noun_chunks_top10.png")
plt.show()
