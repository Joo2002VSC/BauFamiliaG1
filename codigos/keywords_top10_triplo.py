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

# Construir DataFrame com as contagens
df_keywords = pd.DataFrame({
    nome: [contagens_keywords[nome].get(kw, 0) for kw in top_keywords]
    for nome in entrevistas
}, index=top_keywords)

# Ordenar entrevistados pela soma total para o "waterfall"
somas = df_keywords.sum(axis=0).sort_values(ascending=False)
ordenados = somas.index.tolist()
df_keywords = df_keywords[ordenados]

# Calcular as diferenças incrementais (camadas do gráfico waterfall)
df_diff = df_keywords.copy()
for i in range(len(ordenados) - 1, 0, -1):
    df_diff.iloc[:, i] -= df_diff.iloc[:, i - 1]

# Salvar CSV com as diferenças incrementais
df_diff.to_csv("keywords_top10_por_entrevistado.csv", encoding="utf-8")

# Mapa de cores correto
cores = {"Joana": "purple", "João": "blue", "Mariana": "pink"}

# Gráfico waterfall por keyword
x = np.arange(len(df_keywords.index))
bar_width = 0.6

plt.figure(figsize=(12, 6))

bottom = np.zeros(len(df_keywords.index))
for nome in ordenados:
    valores = df_diff[nome].clip(lower=0)
    plt.bar(x, valores, bottom=bottom, label=nome, color=cores[nome])
    bottom += valores

plt.xticks(x, df_keywords.index, rotation=45)
plt.title("Top 10 Keywords mais Frequentes por Entrevistado")
plt.xlabel("Keyword")
plt.ylabel("Frequência Incremental")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("keywords_top10_triplo.png")
plt.show()