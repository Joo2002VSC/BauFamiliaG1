import pandas as pd
import matplotlib.pyplot as plt
import spacy

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Caminho do arquivo
caminho = r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_entrevistas.csv"

# Ler o CSV
df = pd.read_csv(caminho)

# Concatenar todas as respostas (assumindo que a coluna se chama "Resposta")
texto = " ".join(df["Resposta"].fillna(""))

# Processar texto com spaCy
doc = nlp(texto)
palavras = [token.text.lower() for token in doc if token.is_alpha]
total = len(palavras)
unicas = len(set(palavras))
riqueza = unicas / total if total > 0 else 0

# Salvar riqueza lexical em um CSV
df_riqueza = pd.DataFrame({"Entrevista": ["Entrevista"], "Riqueza Lexical": [round(riqueza, 4)]})
df_riqueza.to_csv("riqueza_lexical.csv", index=False)

# Gráfico de barras (apenas um valor)
plt.figure(figsize=(4, 6))
plt.bar(["Entrevista"], [riqueza], color="orange")
plt.title("Riqueza Lexical do Corpus")
plt.ylabel("Riqueza Lexical")
plt.ylim(0, 0.5)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("riqueza_lexical.png")
plt.show()