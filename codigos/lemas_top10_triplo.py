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
os.makedirs("graficos_lemas_individuais", exist_ok=True)

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

    lemas = [
        token.lemma_.lower() for token in doc
        if token.is_alpha
        and not token.is_stop
        and len(token.text) > 2
    ]

    contagem = Counter(lemas)
    top_10 = contagem.most_common(10)

    # Criar DataFrame e guardar CSV
    df_top = pd.DataFrame(top_10, columns=["Lema", "Frequência"])
    df_top.to_csv(f"lemas_top10_{nome}.csv", index=False, encoding="utf-8")

    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(df_top["Lema"], df_top["Frequência"], color=cores[nome])
    plt.title(f"Top 10 Lemas - {nome}")
    plt.xlabel("Lema")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graficos_lemas_individuais/lemas_top10_{nome}.png")
    plt.close()