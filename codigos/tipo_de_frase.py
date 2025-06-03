import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter

# Carregar modelo spaCy
nlp = spacy.load("pt_core_news_lg")

# Carregar dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_entrevistas.csv")

# Concatenar perguntas e respostas
texto = " ".join(df["Pergunta"].fillna("") + " " + df["Resposta"].fillna(""))

# Processar o texto com spaCy
doc = nlp(texto)

# Função para classificar o tipo de frase
def classificar_frase(sent):
    texto = sent.text.strip()
    if texto.endswith("?"):
        return "Interrogativa"
    elif texto.endswith("!"):
        return "Exclamativa"
    elif any(token.tag_ == "VERB" and "Imp" in token.morph.get("Mood") for token in sent):
        return "Imperativa"
    else:
        return "Declarativa"

# Classificar todas as frases
tipos_frases = [(sent.text.strip(), classificar_frase(sent)) for sent in doc.sents]

# Criar DataFrame com frases e tipos
df_tipos = pd.DataFrame(tipos_frases, columns=["Frase", "Tipo de Frase"])

# Contagem para gráfico
contagem = df_tipos["Tipo de Frase"].value_counts().reset_index()
contagem.columns = ["Tipo de Frase", "Frequência"]

# Exportar CSVs
df_tipos.to_csv("tipos_de_frases_classificadas.csv", index=False)
contagem.to_csv("tipos_de_frase_frequencia.csv", index=False)

# Gráfico
plt.figure(figsize=(8, 5))
plt.bar(contagem["Tipo de Frase"], contagem["Frequência"], color="orange")
plt.title("Distribuição dos Tipos de Frase")
plt.xlabel("Tipo de Frase")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("tipos_de_frase.png")
plt.show()
