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

# Filtros
tipos_validos = {"PER", "LOC", "ORG", "DATE", "GPE"}
palavras_excluir = {
    "Tarefa", "Difícil", "Campo", "Humildade", "Memórias", "tempestade", "Dávamo-nos", "Podiam", "Claro","Setembro"
}

# Criar diretório para resultados
os.makedirs("graficos_entidades_individuais", exist_ok=True)

for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)
    texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

    doc = nlp(texto)

    entidades = [
        ent.text.strip() for ent in doc.ents
        if ent.label_ in tipos_validos
        and ent.text.strip() not in palavras_excluir
        and ent.text.strip()[0].isupper()
    ]

    contagem = Counter(entidades)
    top_10 = contagem.most_common(10)

    # Criar DataFrame e guardar
    df_top = pd.DataFrame(top_10, columns=["Entidade", "Frequência"])
    df_top.to_csv(f"entidades_top10_{nome}.csv", index=False)

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(df_top["Entidade"], df_top["Frequência"])
    plt.title(f"Top 10 Entidades Nomeadas - {nome}")
    plt.xlabel("Entidade")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"graficos_entidades_individuais/entidades_top10_{nome}.png")
    plt.close()