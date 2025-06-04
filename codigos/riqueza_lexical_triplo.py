import pandas as pd
import spacy
import matplotlib.pyplot as plt

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Dicionário com caminhos de cada entrevista
entrevistas = {
    "Joana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joana.csv",
    "João": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joao.csv",
    "Mariana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_mariana.csv"
}

# Cores para cada entrevistado
cores = {
    "Joana": "purple",
    "João": "blue",
    "Mariana": "pink"
}

# Armazenar resultados
resultados = {}

for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)

    # Combina perguntas e respostas
    perguntas = df["Pergunta"].fillna("").tolist()
    respostas = df["Resposta"].fillna("").tolist()
    todas_frases = perguntas + respostas
    texto = " ".join(todas_frases)

    doc = nlp(texto)

    declarativas = 0
    interrogativas = 0

    for sent in doc.sents:
        if sent.text.strip().endswith("?"):
            interrogativas += 1
        else:
            declarativas += 1

    resultados[nome] = {
        "Declarativa": declarativas,
        "Interrogativa": interrogativas
    }

# Converter para DataFrame
df = pd.DataFrame(resultados).T

# Ordenar por frequência cumulativa crescente (melhor visualmente)
ordem = df.sum(axis=1).sort_values().index.tolist()
df = df.loc[ordem]

# Gerar gráfico de diferença acumulada
fig, ax = plt.subplots(figsize=(8, 6))

tipos = ["Declarativa", "Interrogativa"]
base = {tipo: 0 for tipo in tipos}

for nome in df.index:
    for tipo in tipos:
        valor = df.loc[nome, tipo]
        if valor > 0:
            ax.bar(tipo, valor, bottom=base[tipo], color=cores[nome], label=nome if tipo == "Declarativa" else "")
            base[tipo] += valor

# Configurações do gráfico
ax.set_title("Distribuição Acumulada por Tipo de Frase")
ax.set_ylabel("Número de Frases")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Legenda sem duplicados
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), title="Entrevistado")

plt.tight_layout()
plt.savefig("tipos_de_frase_triplo.png")
plt.show()
