import pandas as pd
import matplotlib.pyplot as plt
import spacy

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Caminhos das entrevistas (mantém os nomes originais)
entrevistas = {
    "Joana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joana.csv",
    "João": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joao.csv",
    "Mariana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_mariana.csv"
}

# Mapeamento dos nomes para os nomes dos entrevistados
nomes_grafico = {
    "Joana": "Abel",
    "João": "Rosa",
    "Mariana": "Laurinda"
}

# Dicionário para armazenar os valores
riqueza_lexical = {}

# Calcular riqueza lexical por entrevistado
for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)
    texto = " ".join(df["Resposta"].fillna(""))
    doc = nlp(texto)

    palavras = [token.text.lower() for token in doc if token.is_alpha]
    total = len(palavras)
    unicas = len(set(palavras))

    riqueza = unicas / total if total > 0 else 0
    # Usa o nome do entrevistado para o gráfico
    riqueza_lexical[nomes_grafico[nome]] = round(riqueza, 4)  # Quatro casas decimais

# Criar DataFrame
df_riqueza = pd.DataFrame.from_dict(riqueza_lexical, orient="index", columns=["Riqueza Lexical"])
df_riqueza.to_csv("riqueza_lexical_por_entrevistado.csv")

# Gráfico de barras separadas
plt.figure(figsize=(8, 5))
cores = {"Abel": "purple", "Rosa": "blue", "Laurinda": "pink"}
plt.bar(df_riqueza.index, df_riqueza["Riqueza Lexical"], color=[cores[nome] for nome in df_riqueza.index])
plt.title("Riqueza Lexical por Entrevistado")
plt.xlabel("Entrevistado")
plt.ylabel("Riqueza Lexical")
plt.ylim(0, 0.5)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("riqueza_lexical_tripo.png")
plt.show()