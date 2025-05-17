tudo: entrevista_ana_final.conll entrevista_ana_final-ent.txt

# Anotar o texto 
entrevista_ana_final.conll: entrevista_ana_final.txt
	avdconll entrevista_ana_final.txt > entrevista_ana_final.conll

# Calcular as entidades
entrevista_ana_final-ent.txt: entrevista_ana_final.conll
	rg PROPN entrevista_ana_final.conll | sort | uniq -c | sort -n > entrevista_ana_final-ent.txt

# Calcular as keywords
entrevista_ana_final-keywords.txt: entrevista_ana_final.txt
	keyterms entrevista_ana_final.txt | sort | uniq -c | sort -n > entrevista_ana_final-keywords.txt

# Calcular o nome mais frequente
