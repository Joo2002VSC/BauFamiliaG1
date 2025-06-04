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
    # Combinar pergunta e resposta
    texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))
    doc = nlp(texto)
    
    # Contar verbos (lematizados, minúsculos, sem stopwords, apenas letras)
    verbos = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ == "VERB" and not token.is_stop and token.is_alpha
    ]
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

# Gráfico de barras lado a lado
x = np.arange(len(df_verbos.index))  # posições dos verbos
largura = 0.25  # largura de cada barra

plt.figure(figsize=(12, 6))

# Plotar cada entrevistado com deslocamento no eixo X
for i, nome in enumerate(df_verbos.columns):
    plt.bar(x + i * largura, df_verbos[nome], width=largura, label=nome,
            color={"Joana": "purple", "João": "blue", "Mariana": "pink"}[nome])

plt.xticks(x + largura, df_verbos.index, rotation=45)
plt.title("Top 10 Verbos - Frequência por Entrevistado")
plt.xlabel("Verbo (lema)")
plt.ylabel("Frequência")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("top10_verbos_triplo.png")
plt.show()
