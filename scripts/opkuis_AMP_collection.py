import os
import pandas as pd

script_dir = os.path.dirname(__file__)

AMP_collection = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_collection.csv'))

#general cleanup
AMP_collection['Description'] = AMP_collection['Description'].str.lower()

##typfouten
AMP_collection['Description'] = AMP_collection['Description'].str.replace('intitial', 'initial', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' initial', 'initial', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('initital ', 'initial ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('weightat birth', 'weight at', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('waening', 'weaning', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('atb ', 'at ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' st ', ' at ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('piberty', 'puberty', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('totasl ', '', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('lengtb ', 'length ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('lenght', 'length', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('  ', ' ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('at at', 'at', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('bith', 'birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(r'\birth\b', 'birth', regex=False)
#AMP_collection['Description'] = AMP_collection['Description'].str.replace('  ', ' ', regex=False) 


##terminologie
AMP_collection['Description'] = AMP_collection['Description'].str.replace('males', 'male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('females','female', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('juveniles', 'juvenile', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('adults', 'adult', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('eggs', 'egg', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('at hatching', 'at hatch', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('at hatch', 'at birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('fem.', 'female', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('(female)', 'female', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('(male)', 'male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(r'\bfem\b', 'female', regex=True)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('fmale', 'female', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' of ', ' ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' for ', ' ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('max ', 'maximum ', regex=False)


#life span secific cleanup
AMP_collection['Description'] = AMP_collection['Description'].str.replace('life span', 'lifespan', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('adult lifespan', 'lifespan', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('maximum lifespan', 'lifespan', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('mean lifespan', 'lifespan', regex=False)

#weight specific cleanup
AMP_collection['Description'] = AMP_collection['Description'].str.replace('weigh ', 'weight ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('max wet weight', 'ultimate wet weight', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' (g)', '', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('wet egg weight', 'weight egg', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('weight hatchlings', 'weight at birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('wet dry', 'wet', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('weight birth', 'weight at birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('female weight', 'weight female', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('male weight male', 'weight male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('male weight', 'weight male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('final wet weight', 'ultimate weight', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('ultimate weight female (at svl 7.680 cm)', 'ultimate weight female', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('ultimate weight male (at svl 8.097 cm)', 'ultimate weight male', regex=False)

AMP_collection['Description'] = AMP_collection['Description'].str.replace('wet weight', 'weight', regex=False)


#reproduction rate specific cleanup
AMP_collection['Description'] = AMP_collection['Description'].str.replace('reprod ', 'reproduction ', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('weight female at first reprod', 'weight female at first reproduction', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('time since birth at first reprod', 'time since birth at first reproduction', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('ultimate reproduction rate', 'maximum reproduction rate', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('maximum reproduction rate ', 'maximum reproduction rate', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('reproduction rate maximum rate', 'maximum reproduction rate', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('maximum reproduction ratemaximum rate', 'maximum reproduction rate', regex=False)

#total length specific cleanup
AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length at birth tadpoles', 'total length at birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length tadpole at birth', 'total length at birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length adult', 'ultimate total length', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' total length at puberty', 'total length at puberty', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace(' total length at birth', 'total length at birth', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length at puberty  males', 'total length at puberty male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].apply(
    lambda x: 'ultimate total length male' if x == 'total length male' else x
)
AMP_collection['Description'] = AMP_collection['Description'].apply(
    lambda x: 'ultimate total length female' if x == 'total length female' else x
)
AMP_collection['Description'] = AMP_collection['Description'].apply(
    lambda x: 'total length' if x == ' total length' else x
)

AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length puberty', 'total length at puberty', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length puberty male', 'total length at puberty male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('total length at puberty in male', 'total length at puberty male', regex=False)



#time since birth at puberty specific cleanup
AMP_collection['Description'] = AMP_collection['Description'].apply(
    lambda x: 'time since birth at puberty' if x == 'time since birth at pubert' else x
)

AMP_collection['Description'] = AMP_collection['Description'].str.replace('time since birth at puberty in male', 'time since birth at puberty male', regex=False)
AMP_collection['Description'] = AMP_collection['Description'].str.replace('time since hatch at puberty', 'time since birth at puberty', regex=False)


unique_values = sorted(AMP_collection['Description'].unique())

description_counts = AMP_collection['Description'].value_counts()

# Create a DataFrame with sorted unique values and their counts
unique_values_with_counts = pd.DataFrame({
    'Description': unique_values,
    'Count': [description_counts[desc] for desc in unique_values]
})

AMP_collection.to_csv(os.path.join(script_dir, '../csv_files/AMP_collection_cleaned.csv'), index=False)
