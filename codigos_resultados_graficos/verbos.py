import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Carregar modelo spaCy em português
nlp = spacy.load("pt_core_news_lg")

# Carregar dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos_resultados_graficos\dataset_entrevistas.csv")

# Concatenar todas as perguntas e respostas
texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar com spaCy
doc = nlp(texto)

# Extrair verbos lematizados (sem stopwords)
verbos = [
    token.lemma_.lower()
    for token in doc
    if token.pos_ == "VERB" and not token.is_stop and token.is_alpha
]

# Contar frequência
frequencias = Counter(verbos)

# Converter em DataFrame e guardar CSV
df_verbos = pd.DataFrame(frequencias.items(), columns=["Verbo", "Frequência"])
df_verbos = df_verbos.sort_values(by="Frequência", ascending=False)
df_verbos.to_csv("verbos_frequencia.csv", index=False)

# Gerar gráfico Top 10
top_10 = df_verbos.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10["Verbo"], top_10["Frequência"], color="orange")
plt.title("Top 10 Verbos mais Frequentes")
plt.xlabel("Verbo")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("verbos_top10.png")
plt.show()
