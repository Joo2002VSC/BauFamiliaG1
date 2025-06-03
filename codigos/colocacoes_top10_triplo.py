import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import os

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Caminhos dos ficheiros
entrevistas = {
    "Joana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joana.csv",
    "João": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joao.csv",
    "Mariana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_mariana.csv"
}

# Cores para o gráfico
cores = {"Joana": "purple", "João": "blue", "Mariana": "pink"}

# Criar diretório para resultados
os.makedirs("graficos_colocacoes_individuais", exist_ok=True)

for nome, caminho in entrevistas.items():
    if not os.path.isfile(caminho):
        print(f"⚠️ Ficheiro não encontrado para {nome}: {caminho}")
        continue

    try:
        df = pd.read_csv(caminho, encoding="utf-8")
    except Exception as e:
        print(f"❌ Erro ao ler o ficheiro de {nome}: {e}")
        continue

    texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

    doc = nlp(texto)

    colocacoes = []

    for i in range(len(doc) - 1):
        token1 = doc[i]
        token2 = doc[i + 1]
        
        if (
            token1.is_alpha and token2.is_alpha and
            not token1.is_stop and not token2.is_stop and
            len(token1.text) > 2 and len(token2.text) > 2
        ):
            if (
                (token1.pos_ == "VERB" and token2.pos_ == "NOUN") or
                (token1.pos_ == "NOUN" and token2.pos_ == "NOUN")
            ):
                colocacao = f"{token1.lemma_} {token2.lemma_}"
                colocacoes.append(colocacao.lower())

    contagem = Counter(colocacoes)
    top_10 = contagem.most_common(10)

    # Criar DataFrame e guardar CSV
    df_top = pd.DataFrame(top_10, columns=["Colocação", "Frequência"])
    df_top.to_csv(f"colocacoes_top10_{nome}.csv", index=False, encoding="utf-8")

    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(df_top["Colocação"], df_top["Frequência"], color=cores[nome])
    plt.title(f"Top 10 Colocações - {nome}")
    plt.xlabel("Colocação")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graficos_colocacoes_individuais/colocacoes_top10_{nome}.png")
    plt.close()