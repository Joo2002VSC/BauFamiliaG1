import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Caminhos dos ficheiros
entrevistas = {
    "Joana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joana.csv",
    "João": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joao.csv",
    "Mariana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_mariana.csv"
}

# Dicionário para armazenar contagens de verbos por entrevistado
verbos_por_entrevistado = {}

# Processar cada entrevista
for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)
    texto = " ".join(df["Resposta"].fillna(""))
    doc = nlp(texto)
    
    # Contar verbos (lematizados e em minúsculas)
    verbos = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
    contagem = Counter(verbos)
    verbos_por_entrevistado[nome] = contagem

# Identificar os 10 verbos mais frequentes no total (soma de todos)
todos_verbos = sum(verbos_por_entrevistado.values(), Counter())
top_10_verbo = [verbo for verbo, _ in todos_verbos.most_common(10)]

# Criar DataFrame com contagens dos 10 verbos
df_verbos = pd.DataFrame({
    nome: [verbos_por_entrevistado[nome].get(verbo, 0) for verbo in top_10_verbo]
    for nome in entrevistas
}, index=top_10_verbo)

# Exportar CSV
df_verbos.to_csv("top10_verbos_por_entrevistado.csv", encoding="utf-8")

# Gráfico de barras "acumuladas por diferença"
x = np.arange(len(df_verbos.index))
plt.figure(figsize=(12, 6))

# Ordenar os entrevistados por maior valor para cada verbo
for i, verbo in enumerate(df_verbos.index):
    valores = df_verbos.loc[verbo]
    ordenados = valores.sort_values(ascending=True)
    base = 0
    for nome in ordenados.index:
        altura = ordenados[nome] - base
        if altura > 0:
            cor = {"Joana": "purple", "João": "blue", "Mariana": "pink"}[nome]
            plt.bar(i, altura, bottom=base, label=nome if i == 0 else "", color=cor)
            base = ordenados[nome]

plt.xticks(x, df_verbos.index, rotation=45)
plt.title("Top 10 Verbos mais Frequentes por Entrevistado")
plt.xlabel("Verbo (lema)")
plt.ylabel("Frequência")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("top10_verbos_triplo.png")
plt.show()