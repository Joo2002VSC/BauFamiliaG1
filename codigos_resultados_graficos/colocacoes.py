import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Carregar o dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos_resultados_graficos\dataset_entrevistas.csv")

# Concatenar perguntas e respostas
texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar com spaCy
doc = nlp(texto)

# Extrair tokens válidos (alfabéticos e não stopwords)
tokens_validos = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]

# Gerar bigramas (pares de palavras consecutivas)
bigramas = list(zip(tokens_validos, tokens_validos[1:]))

# Contar frequência de bigramas
frequencias = Counter(bigramas)

# Converter para DataFrame
df_bigrams = pd.DataFrame(frequencias.items(), columns=["Bigrama", "Frequência"])
df_bigrams["Bigrama"] = df_bigrams["Bigrama"].apply(lambda x: " ".join(x))
df_bigrams = df_bigrams.sort_values(by="Frequência", ascending=False)

# Exportar para CSV
df_bigrams.to_csv("colocacoes_frequencia.csv", index=False)

# Gráfico Top 10 com cor laranja
top_10 = df_bigrams.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10["Bigrama"], top_10["Frequência"], color="orange")
plt.title("Top 10 Colocações mais Frequentes")
plt.xlabel("Colocações")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("colocacoes_top10.png")
plt.show()