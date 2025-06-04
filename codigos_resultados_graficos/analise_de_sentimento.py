import pandas as pd
import matplotlib.pyplot as plt
import re

# Listas simples de palavras positivas e negativas
palavras_positivas = {
    "feliz", "espetacular", "gostava", "divertido", "dançar", "carinho", "alegria",
    "brincar", "união", "experiências", "cultura", "recordam", "solidária", "liberdade",
    "cresceu", "qualidade", "respeito", "humildade", "marcantes", "apoio", "ajudar",
    "sorriso", "brincadeiras", "amizade", "sonho", "agradável", "calma", "bola", "macaca", 
    "escondidas", "malha", "domingos", "professora", "gerência", "cargo", "natureza", "avó", 
    "irmãos", "avô", "pai","marinheiro", "Salvou", "praias", "bicicleta","Marinha", "únicas", 
    "pais", "avós", "familiar", "rica", "maior", "agilidade"
}

palavras_negativas = {
    "difícil", "brigas", "obrigado", "duras", "castigo", "severa", "fome", "triste",
    "problema", "dificuldades", "menos", "parar", "não", "reserva", "escondiam",
    "cuidar", "trabalho", "cansada", "sofrido", "rigorosa", "castigava", "pesado", "pequena", 
    "Pouco", "tarefas", "difíceis", "desfolhar", "sachar", "fazer tudo","reserva"
}

# Função para classificar sentimento com base nas listas
def classificar_sentimento(texto):
    if not isinstance(texto, str) or texto.strip() == "":
        return "Neutro"

    # Normalizar o texto (minúsculas, sem pontuação)
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

# Carregar dataset
df = pd.read_csv(r"C:\Users\joana\Desktop\Joana\Universidade\UMinho\HD\AVD\BauFamiliaG1\codigos_resultados_graficos\dataset_entrevistas.csv")

# Aplicar a função a cada resposta
# Aplica apenas a respostas não vazias
df["Sentimento"] = df["Resposta"].apply(lambda r: classificar_sentimento(r) if isinstance(r, str) and r.strip() else "")

# Guardar novo CSV
df.to_csv("analise_de_sentimentos.csv", index=False)

# Gráfico da distribuição
# Filtrar apenas linhas com sentimento válido
df_filtrado = df[df["Sentimento"].isin(["Positivo", "Negativo", "Neutro"])]

# Contar quantos há de cada tipo
contagem = df_filtrado["Sentimento"].value_counts()

# Cores atribuídas com base no índice
cores = {
    "Positivo": "green",
    "Neutro": "gray",
    "Negativo": "red"
}

# Plot
plt.figure(figsize=(6, 4))
contagem.plot(kind="bar", color=[cores.get(sent, "blue") for sent in contagem.index])
plt.title("Análise de Sentimentos nas Respostas")
plt.xlabel("Sentimento")
plt.ylabel("Respostas")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("analise_de_sentimentos.png")
plt.show()