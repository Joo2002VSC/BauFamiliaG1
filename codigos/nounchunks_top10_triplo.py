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
os.makedirs("graficos_chunks_individuais", exist_ok=True)

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

    chunks = [
        chunk.text.strip().lower()
        for chunk in doc.noun_chunks
        if len(chunk.text.strip()) > 2
    ]

    contagem = Counter(chunks)
    top_10 = contagem.most_common(10)

    # Criar DataFrame e guardar CSV
    df_top = pd.DataFrame(top_10, columns=["Noun Chunk", "Frequência"])
    df_top.to_csv(f"noun_chunks_top10_{nome}.csv", index=False, encoding="utf-8")

    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(df_top["Noun Chunk"], df_top["Frequência"], color=cores[nome])
    plt.title(f"Top 10 Noun Chunks - {nome}")
    plt.xlabel("Noun Chunk")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graficos_chunks_individuais/noun_chunks_top10_{nome}.png")
    plt.close()