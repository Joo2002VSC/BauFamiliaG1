import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Ficheiros das entrevistas
entrevistas = {
    "Joana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joana.csv",
    "João": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joao.csv",
    "Mariana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_mariana.csv"
}

# Dicionário para guardar contagens
contagens_keywords = {}

for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)
    texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))
    doc = nlp(texto)

    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "ADJ"}
        and not token.is_stop
        and token.is_alpha
        and len(token.text) > 2
    ]

    contagem = Counter(keywords)
    contagens_keywords[nome] = contagem

# Top 10 keywords mais frequentes no total
todas_keywords = Counter()
for contagem in contagens_keywords.values():
    todas_keywords.update(contagem)
top_keywords = [kw for kw, _ in todas_keywords.most_common(10)]

# Construir DataFrame com as contagens absolutas
df_keywords = pd.DataFrame({
    nome: [contagens_keywords[nome].get(kw, 0) for kw in top_keywords]
    for nome in entrevistas
}, index=top_keywords)

# Salvar CSV com as contagens absolutas
df_keywords.to_csv("keywords_top10_por_entrevistado.csv", encoding="utf-8")

# Cores por entrevistado
cores = {"Joana": "purple", "João": "blue", "Mariana": "pink"}

# Gráfico de barras agrupadas
x = np.arange(len(top_keywords))
total_entrevistados = len(df_keywords.columns)
bar_width = 0.2

plt.figure(figsize=(14, 6))

for i, nome in enumerate(df_keywords.columns):
    plt.bar(x + i * bar_width, df_keywords[nome], width=bar_width, label=nome, color=cores[nome])

plt.xticks(x + bar_width, df_keywords.index, rotation=45)
plt.title("Top 10 Keywords por Entrevistado")
plt.xlabel("Keyword")
plt.ylabel("Frequência")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("keywords_top10_agrupado.png")
plt.show()
