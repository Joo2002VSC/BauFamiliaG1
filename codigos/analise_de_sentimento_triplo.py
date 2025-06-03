import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np

# Listas de palavras
palavras_positivas = {
    "feliz", "espetacular", "gostava", "divertido", "dançar", "carinho", "alegria",
    "brincar", "união", "experiências", "cultura", "recordam", "solidária", "liberdade",
    "cresceu", "qualidade", "respeito", "humildade", "marcantes", "apoio", "ajudar",
    "sorriso", "brincadeiras", "amizade", "sonho", "agradável", "calma", "bola", "macaca", 
    "escondidas", "malha", "domingos", "professora", "gerência", "cargo", "natureza", "avó", 
    "irmãos", "avô", "pai", "marinheiro", "salvou", "praias", "bicicleta", "marinha", "únicas", 
    "pais", "avós", "familiar", "rica", "maior", "agilidade"
}

palavras_negativas = {
    "difícil", "brigas", "obrigado", "duras", "castigo", "severa", "fome", "triste",
    "problema", "dificuldades", "menos", "parar", "não", "reserva", "escondiam",
    "cuidar", "trabalho", "cansada", "sofrido", "rigorosa", "castigava", "pesado", "pequena", 
    "pouco", "tarefas", "difíceis", "desfolhar", "sachar", "fazer tudo", "reserva"
}

# Função para classificar sentimento
def classificar_sentimento(texto):
    if not isinstance(texto, str) or texto.strip() == "":
        return "Neutro"

    palavras = re.findall(r'\b\w+\b', texto.lower())
    score = 0
    for palavra in palavras:
        if palavra in palavras_positivas:
            score += 1
        elif palavra in palavras_negativas:
            score -= 1

    if score > 0:
        return "Positivo"
    elif score < 0:
        return "Negativo"
    else:
        return "Neutro"

# Dicionário com caminhos
entrevistas = {
    "Joana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joana.csv",
    "João": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_joao.csv",
    "Mariana": r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos\dataset_corpus_entrevista_mariana.csv"
}

# Armazenar resultados
resultados = {}

for nome, caminho in entrevistas.items():
    df = pd.read_csv(caminho)
    sentimentos = df["Resposta"].apply(classificar_sentimento)
    contagem = Counter(sentimentos)
    resultados[nome] = contagem

# Sentimentos a analisar
todos_sentimentos = ["Negativo", "Neutro", "Positivo"]

# Criar DataFrame com os dados
df_sentimentos = pd.DataFrame(
    {nome: [resultados[nome].get(sent, 0) for sent in todos_sentimentos] for nome in entrevistas},
    index=todos_sentimentos
)

# Exportar CSV
df_sentimentos.to_csv("analise_de_sentimentos_por_entrevistado.csv", encoding="utf-8")

# Gráfico de barras lado a lado
x = np.arange(len(todos_sentimentos))
largura = 0.25  # Largura das barras

plt.figure(figsize=(10, 6))

plt.bar(x - largura, df_sentimentos["Joana"], width=largura, label="Joana", color="purple")
plt.bar(x, df_sentimentos["João"], width=largura, label="João", color="blue")
plt.bar(x + largura, df_sentimentos["Mariana"], width=largura, label="Mariana", color="pink")

plt.xticks(x, todos_sentimentos)
plt.title("Análise de Sentimentos por Entrevistado")
plt.xlabel("Sentimento")
plt.ylabel("Número de Respostas")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("analise_de_sentimentos_barras_lado_a_lado.png")
plt.show()
