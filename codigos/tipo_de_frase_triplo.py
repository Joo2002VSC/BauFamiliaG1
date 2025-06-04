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

# Função para classificar tipo de frase
def classificar_frase(sent):
    texto = sent.text.strip()
    if texto.endswith("?"):
        return "Interrogativa"
    elif texto.endswith("!"):
        return "Exclamativa"
    elif any(token.tag_ == "VERB" and "Imp" in token.morph.get("Mood") for token in sent):
        return "Imperativa"
    else:
        return "Declarativa"

# Dicionário para guardar resultados
resultados = {}

# Analisar cada entrevista (pergunta + resposta)
for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)
    texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))
    doc = nlp(texto)
    tipos_frases = [classificar_frase(sent) for sent in doc.sents]
    contagem = Counter(tipos_frases)
    resultados[nome] = contagem

# Obter todos os tipos de frase e ordenar
todos_tipos = sorted(set(tipo for cont in resultados.values() for tipo in cont))

# Criar DataFrame com os dados organizados
df_resultados = pd.DataFrame(
    {nome: [resultados[nome].get(tipo, 0) for tipo in todos_tipos] for nome in resultados},
    index=todos_tipos
)

# Exportar CSV
df_resultados.to_csv("tipos_de_frase_por_entrevistado.csv", encoding="utf-8")

# Gráfico de barras lado a lado
x = np.arange(len(df_resultados.index))  # posições dos tipos
largura = 0.25  # largura das barras

plt.figure(figsize=(10, 6))

for i, nome in enumerate(df_resultados.columns):
    plt.bar(x + i * largura, df_resultados[nome], width=largura, label=nome,
            color={"Joana": "purple", "João": "blue", "Mariana": "pink"}[nome])

plt.xticks(x + largura, df_resultados.index)
plt.title("Distribuição dos Tipos de Frase por Entrevistado")
plt.xlabel("Tipo de Frase")
plt.ylabel("Frequência")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("tipos_de_frase_triplo")
plt.show()
