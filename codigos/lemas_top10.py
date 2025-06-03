import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Carregar o dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_entrevistas.csv")

# Concatenar perguntas e respostas
texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar com spaCy
doc = nlp(texto)

# Extrair lemas (sem stopwords, só texto alfabético)
lemas = [
    token.lemma_.lower()
    for token in doc
    if token.is_alpha and not token.is_stop
]

# Contar frequência
frequencias = Counter(lemas)

# Converter em DataFrame e guardar CSV
df_lemas = pd.DataFrame(frequencias.items(), columns=["Lema", "Frequência"])
df_lemas = df_lemas.sort_values(by="Frequência", ascending=False)
df_lemas.to_csv("lemas.csv", index=False)

# Gerar gráfico Top 10
top_10 = df_lemas.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10["Lema"], top_10["Frequência"], color="orange")
plt.title("Top 10 Lemas mais Frequentes")
plt.xlabel("Lema")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("lemas_top10.png")
plt.show()
