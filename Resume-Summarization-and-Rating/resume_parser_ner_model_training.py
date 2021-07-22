#*******************************************************************************************************************************************************
## Loading Libraries

import warnings
import json
import random
import logging
import spacy
import re

warnings.filterwarnings("ignore")


#*******************************************************************************************************************************************************
## Creating blank Language class
nlp = spacy.blank('en')  


#*******************************************************************************************************************************************************
## Taking input for the Directory path in order to save the trained model
dir_path = input("Enter the Directory path to save the trained model: ")

if(dir_path[-1] != '/'):
    dir_path = dir_path + '/'

print("\nDirectory path is :: ",dir_path)


#*******************************************************************************************************************************************************
def convert_data_to_spacy(JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content'].replace("\n", " ")
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        return training_data

    except Exception as e:
        logging.exception("Unable to process " + JSON_FilePath + "\n" + "error = " + str(e))
        return None


#*******************************************************************************************************************************************************
## Function for removing leading and trailing white spaces from entity spans

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data


#*******************************************************************************************************************************************************
## Training Spacy Name Entity Recognition model

def train_spacy():

    TRAIN_DATA = trim_entity_spans(convert_data_to_spacy("traindata.json"))
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                except Exception as e:
                    pass
            print(losses)
            

#*******************************************************************************************************************************************************

## Calling the Training function     
train_spacy()

## Saving the trained model
nlp.to_disk(dir_path + 'nlp_model')