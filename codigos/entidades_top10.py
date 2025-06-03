import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Palavras que queremos excluir por erro frequente
palavras_excluir = {
    "Tarefa", "Difícil", "Campo", "Humildade", "Memórias", "tempestade", "Dávamo-nos", "Podiam"
}

# Tipos de entidade permitidos
tipos_validos = {"PER", "LOC", "ORG", "DATE", "GPE"}

# Carregar o dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_entrevistas.csv")

texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar com spaCy
doc = nlp(texto)

# Extrair entidades filtradas
entidades_filtradas = [
    (ent.text.strip(), ent.label_)
    for ent in doc.ents
    if ent.label_ in tipos_validos
    and ent.text.strip() not in palavras_excluir
    and ent.text.strip()[0].isupper()
]

# Contar frequência
frequencias = Counter(entidades_filtradas)

# Converter para DataFrame
df_entidades = pd.DataFrame(
    [(texto, tipo, freq) for (texto, tipo), freq in frequencias.items()],
    columns=["Entidade", "Tipo", "Frequência"]
)
df_entidades = df_entidades.sort_values(by="Frequência", ascending=False)

# Guardar CSV
df_entidades.to_csv("entidades_top10.csv", index=False)

# Gráfico Top 10 Entidades (em laranja)
top_10 = df_entidades.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10["Entidade"], top_10["Frequência"], color="orange")
plt.title("Top 10 Entidades mais Frequentes")
plt.xlabel("Entidade")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("entidades_top10.png")
plt.show()
