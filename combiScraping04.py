# Importar los paquetes necesarios
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from spacy import displacy

# Especificar la URL
#url = 'https://www.python.org/~guido/'
#url='https://www.thetimes.co.uk/'

url='https://www.nasa.gov/'
# Realizar la solicitud y obtener la respuesta
response = requests.get(url)

# Extraer el HTML de la respuesta
html_doc = response.text

# Crear un objeto BeautifulSoup para el análisis
soup = BeautifulSoup(html_doc, 'html.parser')

# Extraer el texto del HTML
text_content = soup.get_text()

# Función para cargar el contenido del artículo
def load_article(article_content):
    # Retorna el contenido del artículo
    return article_content

# Llamar a la función load_article con el texto extraído
article = load_article(text_content)

# Cargar el modelo de inglés de spaCy
nlp = spacy.load('en_core_web_sm')

# Crear un nuevo documento spaCy con el contenido del artículo
doc = nlp(article)

# Obtener las palabras y contar su frecuencia
words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
word_freq = Counter(words)

# Obtener las 10 palabras más frecuentes
top_10_words = word_freq.most_common(10)

# Imprimir las 10 palabras más frecuentes y su frecuencia
print("Las 10 palabras más frecuentes y su frecuencia:")
for word, freq in top_10_words:
    print(f"{word}: {freq}")

# Crear un DataFrame para el gráfico de barras
df_top_words = pd.DataFrame(top_10_words, columns=['Palabra', 'Frecuencia'])

# Graficar las 10 palabras más frecuentes
plt.figure(figsize=(10, 6))
sns.barplot(x='Frecuencia', y='Palabra', data=df_top_words, palette='viridis')
plt.xlabel('Frecuency')
plt.ylabel('Word')
plt.title('Top 10 more frecuent words')
plt.tight_layout()
plt.savefig('top_10_words.png')

# Mostrar el gráfico de barras
plt.show()

# HTML
html = displacy.render(doc, style='ent', page=True)
with open('outputHtmlViewEntitysResults03.html', 'w', encoding='utf-8') as file:
    file.write(html)

# Crear un DataFrame para almacenar las entidades encontradas
entidadesEncontradas = pd.DataFrame(columns=['Texto', 'Categoría'])

# Llenar el DataFrame con las entidades encontradas
for ent in doc.ents:
    entidadesEncontradas = entidadesEncontradas._append({'Texto': ent.text, 'Categoría': ent.label_}, ignore_index=True)

# Contar las ocurrencias de cada categoría
conteo_categorias = entidadesEncontradas['Categoría'].value_counts()

# Graficar
plt.figure(figsize=(10, 6))
conteo_categorias.plot(kind='bar', color=plt.cm.Paired(range(len(conteo_categorias))))
plt.xlabel('Categoría')
plt.ylabel('Cantidad')
plt.title('Conteo de Categorías de Entidades')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot.png')
# Mostrar la gráfica
plt.show()