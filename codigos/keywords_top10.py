import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Carregar modelo spaCy para português
nlp = spacy.load("pt_core_news_lg")

# Carregar dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_entrevistas.csv")

# Juntar perguntas e respostas num só texto
texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar com spaCy
doc = nlp(texto)

# Extrair substantivos e adjetivos (keywords)
keywords = [
    token.lemma_.lower()
    for token in doc
    if token.pos_ in {"NOUN", "ADJ"}
    and not token.is_stop
    and token.is_alpha
    and len(token.text) > 2
]

# Contar frequência
frequencia = Counter(keywords)

# Converter para DataFrame e guardar CSV
df_keywords = pd.DataFrame(frequencia.items(), columns=["Keyword", "Frequência"])
df_keywords = df_keywords.sort_values(by="Frequência", ascending=False)
df_keywords.to_csv("keywords_top10.csv", index=False)

# Gráfico Top 10
top_10 = df_keywords.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10["Keyword"], top_10["Frequência"], color="orange")
plt.title("Top 10 Keywords mais Frequentes")
plt.xlabel("Keyword")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("keywords_top10.png")
plt.show()
