import pandas as pd
import spacy
import matplotlib.pyplot as plt

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Caminho do dataset geral (com todas as entrevistas juntas)
caminho_geral = r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos_resultados_graficos\dataset_entrevistas.csv"

# Ler o CSV
df_geral = pd.read_csv(caminho_geral)

# Concatenar todas as respostas
texto_geral = " ".join(df_geral["Resposta"].fillna(""))

# Processar texto
doc_geral = nlp(texto_geral)

# Filtrar tokens (apenas palavras alfabéticas em minúsculas)
palavras_geral = [token.text.lower() for token in doc_geral if token.is_alpha]

total_geral = len(palavras_geral)
unicas_geral = len(set(palavras_geral))

riqueza_geral = unicas_geral / total_geral if total_geral > 0 else 0

# Salvar CSV
df_saida = pd.DataFrame({
    "Métrica": ["Riqueza Lexical Geral"],
    "Valor": [round(riqueza_geral, 4)]
})
df_saida.to_csv("riqueza_lexical_geral.csv", index=False)

# Gerar gráfico
plt.figure(figsize=(4, 6))
plt.bar(["Geral"], [riqueza_geral], color="orange")
plt.title("Riqueza Lexical do Dataset Geral")
plt.ylabel("Riqueza Lexical")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("grafico_riqueza_geral.png")
plt.show()
