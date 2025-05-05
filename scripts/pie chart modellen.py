import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os

script_dir = os.path.dirname(__file__)

# inlezen nodige data
species = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_species_list.csv'))
species = pd.read_csv(os.path.join(script_dir, '../csv_files/mitofish_sequences.csv'))

# modellen uit data frame halen
models = species[['Mod']]

# het aantal modellen bekijken, tellen hoeveel keer ze voorkomen en sorteren voor estetische overwegingen
counted_models = dict(Counter(models['Mod']))
sorted_models = dict(sorted(counted_models.items(), key=lambda item: item[1], reverse=True))

# de labels in een lijst steken
labels = list(sorted_models.keys())

# de waarden uit de dictionary halen
occurences = []
for key in sorted_models.keys():
    occurences.append(sorted_models[key])

# het percentage van voorkomen in onze dataset van elk model berekenen
total = sum(occurences)
percentage = []
for i in range(len(occurences)):
    percentage.append(100 * occurences[i] / total)

# de legende voorbereiden
label = [f'{l}, {s:0.1f}%' for l, s in zip(labels, percentage)]

# een piechart initiëren en hier onze data aan meegeven 
fig, ax = plt.subplots()
ax.pie(occurences, labels=labels, labeldistance=None)
ax.legend(loc='center left', labels=label, bbox_to_anchor=(-0.3, 0.5))

# Titel toevoegen
ax.set_title('modelverdeling mitofish Dataset', loc='center')

# eventueel opslaan van de figuur 
plt.savefig(os.path.join(script_dir, '../grafieken/piechart_modellen_mitofish.png'), bbox_inches='tight')