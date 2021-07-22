#*******************************************************************************************************************************************************
## Code for checking and installing the required libraries/packages
#*******************************************************************************************************************************************************

import sys
import subprocess
import pkg_resources

required = {'pandas','spacy','nltk','tika','pywebio'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = list(required - installed)

if(missing):
	for lib in missing:
		python = sys.executable
		subprocess.check_call([python, '-m', 'pip', 'install', lib])

# Installing 'en_core_web_sm'
res = subprocess.run(['pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz'], shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)


#*******************************************************************************************************************************************************
## Loading Libraries
#*******************************************************************************************************************************************************

from tika import parser # pip3 install tika 

import pandas as pd
import traceback
import nltk
import spacy
import re

from spacy.matcher import PhraseMatcher
from collections import Counter
from io import StringIO

from pywebio.input import *
from pywebio.output import *

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)


#*******************************************************************************************************************************************************
## Code for taking various inputs (Resume, input/output paths/directories etc.) through GUI
#*******************************************************************************************************************************************************

resume_inputs = input_group("Resume Summary Inputs", [
    input('Enter the Resume directory path', required=True, name='resume_dir_path'),

    file_upload(label='\nUpload Resumes For Parsing', accept=['.pdf','.docx','.doc'], multiple=True, required=True, name='resumes'),

    input('Enter the NLP model directory path', required=True, name='nlp_model_dir_path'),

    input('Enter the keyword file directory path', required=True, name='keywords_dir_path'),

    file_upload(label='Upload Keywords file for getting the keywords score', accept=['.csv'], multiple=False, required=True, name='keywords_file'),

    input('Enter the Output directory path', required=True, name='output_dir_path')

])

skill_inputs = input_group("Mention the Skills priority for resume score calculation", [

    select("Select Year of Experience for consideration", ['0', '1+', '2+','3+', '4+', '5+', '6+', '7+', '8+', '9+', '10+', '12+', '15+', '18+', '20+',
                                                            '2-5', '3-5', '4-7', '5-8', '5-10', '8-10', '10-12', '10-15'], multiple=True, name="experience"),

    radio("Minimum Education Level Required", options=["Bachelor-Masters-PhD", "Masters-PhD", "Bachelor-Masters", "PhD", 
                                                            "Masters", "Bachelor", "Diploma"], required=True, name='education_level'),

    radio("Deep Learning", options=["High", "Medium", "Low", "None"], required=True, name='DL_priority'),

    radio("Machine Learning", options=["High", "Medium", "Low", "None"], required=True, name='ML_priority'),

    radio("Statistics", options=["High", "Medium", "Low", "None"], required=True, name='Stats_priority'),

    radio("Python", options=["High", "Medium", "Low", "None"], required=True, name='Python_priority'),

    # checkbox("statistics", options=['Hypothesis','Analyst'], name='check_box1'),

    # checkbox("Machine Learning", options=[
    #     ('Hypothesis', 1),('Analyst', 2)], name='check_box2')
])

put_text("\n\n\n\n")
put_text("**************************************************************************************************")
put_text("\tPlease wait for the Resume's score to be calculated!!")


#*******************************************************************************************************************************************************
## Variables Initialization based on the inputs from GUI
#*******************************************************************************************************************************************************

## Directory and File path details
resume_dir_path = resume_inputs['resume_dir_path']
resume_details = resume_inputs['resumes']
nlp_model_dir_path = resume_inputs['nlp_model_dir_path']
keywords_dir_path = resume_inputs['keywords_dir_path']
keywords_file = resume_inputs['keywords_file']
output_dir_path = resume_inputs['output_dir_path']


## Skills prioritization details for resume scoring
experience = skill_inputs['experience']
education_level = skill_inputs['education_level']
DL_priority = skill_inputs['DL_priority']
ML_priority = skill_inputs['ML_priority']
Stats_priority = skill_inputs['Stats_priority']
Python_priority = skill_inputs['Python_priority']


## Checking the correctness of directory paths
if(resume_dir_path[-1] != '/'):
    resume_dir_path = resume_dir_path + '/'

if(nlp_model_dir_path[-1] == '/'):
    nlp_model_dir_path = nlp_model_dir_path[:-1]

if(keywords_dir_path[-1] != '/'):
    keywords_dir_path = keywords_dir_path + '/'

if(output_dir_path[-1] != '/'):
    output_dir_path = output_dir_path + '/'


## Preparing list of Resume paths for parsing   
resume_files_list = []
if(type(resume_details) == list):
    for resume in resume_details:
        resume_files_list.append(resume_dir_path + resume['filename'])
        #print(resume['filename'])
else:
    resume_files_list.append(resume_dir_path + resume_details['filename'])
    #print(resume_details['filename'])


## Creating a skill priority dictionary for resume scoring
#skills_priority_dict = {'Years of Experience':experience, 'DL':DL_priority, 'ML':ML_priority, 'Stats':Stats_priority, 'Python':Python_priority}
skills_priority_dict = {'Education Level':education_level, 'DL':DL_priority, 'ML':ML_priority, 'Stats':Stats_priority, 'Python':Python_priority}

for skill, priority in skills_priority_dict.items():
    if(priority == 'High'):
        skills_priority_dict[skill] = 1.0
    elif(priority == 'Medium'):
        skills_priority_dict[skill] = 0.5
    elif(priority == 'Low'):
        skills_priority_dict[skill] = 0.2
    else:
        skills_priority_dict[skill] = 0.0

    

#*******************************************************************************************************************************************************
## Functions for Extracting details from text data
#*******************************************************************************************************************************************************

def tokenization_and_tagging(inputText):

    # Newlines are one element of structure in the data
    # Helps limit the context and breaks up the data as is intended in resumes - i.e., into points

    lines = []
    lines = [el.strip() for el in inputText.split("\n") if len(el) > 0]  # Splitting on the basis of newlines 
    lines = [nltk.word_tokenize(el) for el in lines]    # Tokenize the individual lines
    lines = [nltk.pos_tag(el) for el in lines]  # Tag them

    return lines


#*******************************************************************************************************************************************************
## Function for extracting email from the text
def getEmail(inputText):
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
    #email_pattern = re.compile(r'\S*@\S*')
    email = re.findall(email_pattern, inputText)

    email = list(set(email))
    email = [em.replace('malito:', '').replace('mailto:', '') for em in email]

    return email

    
#*******************************************************************************************************************************************************
## Function for extracting Phone number from the text
def getPhone(inputText):
    '''
    Given an input string, returns possible matches for phone numbers. Uses regular expression based matching.
    Needs an input string, a dictionary where values are being stored, and an optional parameter for debugging.
    Modules required: clock from time, code.
    '''

    number = None
    try:
        pattern = re.compile(r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
            # Understanding the above regex
            # +91 or (91) -> [+(]? \d+ -?
            # Metacharacters have to be escaped with \ outside of character classes; inside only hyphen has to be escaped
            # hyphen has to be escaped inside the character class if you're not incidication a range
            # General number formats are 123 456 7890 or 12345 67890 or 1234567890 or 123-456-7890, hence 3 or more digits
            # Amendment to above - some also have (0000) 00 00 00 kind of format
            # \s* is any whitespace character - careful, use [ \t\r\f\v]* instead since newlines are trouble
        match = pattern.findall(inputText)
        # match = [re.sub(r'\s', '', el) for el in match]
            # Get rid of random whitespaces - helps with getting rid of 6 digits or fewer (e.g. pin codes) strings
        # substitute the characters we don't want just for the purpose of checking
        match = [re.sub(r'[,.]', '', el) for el in match if len(re.sub(r'[()\-.,\s+]', '', el))>6]
            # Taking care of years, eg. 2001-2004 etc.
        match = [re.sub(r'\D$', '', el).strip() for el in match]
            # $ matches end of string. This takes care of random trailing non-digit characters. \D is non-digit characters
        match = [el for el in match if len(re.sub(r'\D','',el)) <= 15]
            # Remove number strings that are greater than 15 digits
        try:
            for el in list(match):
                # Create a copy of the list since you're iterating over it
                if len(el.split('-')) > 3: continue # Year format YYYY-MM-DD
                for x in el.split("-"):
                    try:
                        # Error catching is necessary because of possibility of stray non-number characters
                        # if int(re.sub(r'\D', '', x.strip())) in range(1900, 2100):
                        if x.strip()[-4:].isdigit():
                            if int(x.strip()[-4:]) in range(1900, 2100):
                                # Don't combine the two if statements to avoid a type conversion error
                                match.remove(el)
                    except:
                        pass
        except:
            pass
        number = match
    except:
        pass

    number = list(set(number))
    return number        


#*******************************************************************************************************************************************************
## Function for extracting the Years of Experience from the text
def getExperience(lines):

    experience=[]
    try:
        for sentence in lines:#find the index of the sentence where the degree is find and then analyse that sentence
                sen=" ".join([words[0].lower() for words in sentence]) #string of words in sentence
                if re.search('experience',sen):
                    sen_tokenised= nltk.word_tokenize(sen)
                    tagged = nltk.pos_tag(sen_tokenised)
                    entities = nltk.chunk.ne_chunk(tagged)
                    for subtree in entities.subtrees():
                        for leaf in subtree.leaves():
                            if leaf[1]=='CD':
                                experience=leaf[0]
    except Exception as e:
        print(traceback.format_exc())
        print(e)

    return experience


#*******************************************************************************************************************************************************
## Function that does phrase matching and builds a candidate profile

nlp = spacy.load("en_core_web_sm")

def generate_keyword_summary(text):
    
    text = str(text)
    text = text.lower()
    
    # Below is the csv where we have all the keywords, we can customize it to fit our need or job role
    keyword_dict = pd.read_csv(keywords_dir_path + keywords_file['filename'])

    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
    R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('ML', None, *ML_words)
    matcher.add('DL', None, *DL_words)
    matcher.add('R', None, *R_words)
    matcher.add('Python', None, *python_words)
    matcher.add('DE', None, *Data_Engineering_words)
    doc = nlp(text)
    
    d = []  
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        d.append((rule_id, span.text))

    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    # converting string of keywords to dataframe along with count of keywords
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])

    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    df_final = pd.concat([df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    df_agg = df_final.groupby(['Subject']).size().reset_index(name='counts')
    df_agg = df_agg.rename(columns={"Subject": "Keywords"})

    # Calculating total count of keywords as per the keywords file
    series = keyword_dict.count()
    df_keyword_count = pd.DataFrame({'Keywords':series.index, 'Count':series.values})

    # Mapping for synchronizing the keyword names
    data = {'Keywords':['Statistics', 'NLP', 'Machine Learning', 'Deep Learning', 'R Language', 'Python Language', 'Data Engineering'],
            'Keywords_new':['Stats', 'NLP', 'ML', 'DL', 'R', 'Python', 'DE']}
    df_mapping = pd.DataFrame(data)

    # Mapping old and new keywords name
    df_keyword_final = pd.merge(df_keyword_count,df_mapping, on=['Keywords'], how='left')
    df_keyword_final = df_keyword_final.drop(['Keywords'], axis=1)
    df_keyword_final = df_keyword_final.rename(columns={"Keywords_new": "Keywords", "Count":"Total_Count"})

    # Converting Keywords count to percentage
    df_keyword_percent = pd.merge(df_agg,df_keyword_final, on=['Keywords'], how='left')
    df_keyword_percent['Percent'] = round(df_keyword_percent['counts']/df_keyword_percent['Total_Count'], 2)
    df_keyword_percent = df_keyword_percent.drop(['counts','Total_Count'], axis=1)
    keywords_dict = dict(df_keyword_percent.values)

    # Extracting skills based on keywords
    keyword_list = str(df_final['Keyword'].tolist())
    skills = keyword_list.replace('[','').replace(']','').replace('\'','').replace(':','').strip()

    return(keywords_dict, skills)


#*******************************************************************************************************************************************************
def getResumeScore(candidateDetailsDict, skillsPriorityDict):

    resume_score = 0
    for key in skillsPriorityDict.keys():

        if(key == 'Education Level'):
            if(str(candidateDetailsDict.get(key, 'Bachelor')) in str(skillsPriorityDict[key])):
                resume_score = resume_score + 1.0
            else:
                resume_score = resume_score + 0.5


        # if(key == 'Years of Experience'):
        #     if('-' in skillsPriorityDict[key]):
        #         exp_split = skillsPriorityDict[key].split('-')
        #         if(skillsPriorityDict.get(key, 2) >= candidateDetailsDict[key] and skillsPriorityDict.get(key, 2) <= candidateDetailsDict[key]):
        #             resume_score = resume_score + (skillsPriorityDict.get(key, 0.5) * 1.0)
        #         elif(skillsPriorityDict.get(key, 2) <= candidateDetailsDict[key]):
        #             resume_score = resume_score + (skillsPriorityDict.get(key, 0.5) * 0.5)
        #         else:
        #             resume_score = resume_score + (skillsPriorityDict.get(key, 0.5) * 0.0)
        #     else:
        #         if((skillsPriorityDict.get(key, 2)).replace('+','') >= candidateDetailsDict[key].replace('+',''):
        #             resume_score = resume_score + (skillsPriorityDict.get(key, 0.5) * candidateDetailsDict.get(key, 0.5))

        else:
            resume_score = resume_score + (skillsPriorityDict.get(key, 0.5) * candidateDetailsDict.get(key, 0.5))

    candidateDetailsDict['Resume Score'] = round(resume_score,2)

    return candidateDetailsDict


#*******************************************************************************************************************************************************
## Getting resume text from files using trained nlp model
#*******************************************************************************************************************************************************

# Loading NLP model
nlp_model = spacy.load(nlp_model_dir_path)

# creating an empty dataframe to store the parsed resume data
df = pd.DataFrame()

for file in resume_files_list:

    # parsing file using apache tika parser
    parsed_file = parser.from_file(file)
    parsed_content = parsed_file['content']
    resume_text = parsed_content.replace("\n", " ").replace("\t", " ").strip()

    # Applying nlp model on the test resume(or resumes) 
    resume_doc = nlp_model(resume_text)

    # Extracting entities and corresponding text
    candidate_details = []
    for ent in resume_doc.ents:
        candidate_details.append({ent.label_:ent.text})
        #print(f'{ent.label_.upper():{30}}-{ent.text}')

    # Removing duplicate entities
    candidate_details = [dict(t) for t in {tuple(d.items()) for d in candidate_details}]

    # Saving candidate file name (or resume name) to the dictionary
    file_name = file.split("/")[-1:][0]
    details_dict = {'File Name':file_name}

    # Adding candidate details into one dictionary
    for detail in candidate_details:
        details_dict = {**details_dict, **detail}

    # Extracting Email from the resume text
    if('Email Address' not in details_dict.keys()):
        email_list = getEmail(parsed_content)

        if(email_list):
            details_dict['Email Address'] = email_list[0]

    # Extracting Phone number from the resume text
    if('Phone' not in details_dict.keys()):
        phone_list = getPhone(parsed_content)

        if(phone_list):
            details_dict['Phone'] = phone_list[0]

    # Extracting Years of experience from the resume text
    if('Years of Experience' not in details_dict.keys()):
        lines = tokenization_and_tagging(parsed_content)
        experience = getExperience(lines)

        if(experience):
            details_dict['Years of Experience' ] = str(experience)

    # Adding keywords details to the dictionary
    keywords_dict, skills = generate_keyword_summary(resume_text)
    details_dict.update(keywords_dict)

    # Adding skills if not present
    if('Skills' not in list(details_dict.keys())):
        details_dict['Skills'] = skills

    # Deriving 'Education Level' from 'Degree'
    if('Degree' in details_dict.keys()):
        value = str(details_dict['Degree'])
        if('bachelor' in value.lower() or 'b.tech' in value.lower() or 'b.' in value.lower()):
            details_dict['Education Level'] = 'Bachelor'
        elif('master' in value.lower() or 'm.tech' in value.lower() or 'post' in value.lower() or 'pg' in value.lower() or 'm.' in value.lower()):
            details_dict['Education Level'] = 'Masters'
        elif('diploma' in value.lower()):
            if('post' not in value.lower()):
                details_dict['Education Level'] = 'Diploma'
        elif('phd' in value.lower()):
            details_dict['Education Level'] = 'PhD'

    # Getting resume score for a candidate
    details_dict_with_resume_score = getResumeScore(details_dict, skills_priority_dict)

    # Saving candidate details to the dataframe
    df = df.append(details_dict_with_resume_score, ignore_index=True)


#*******************************************************************************************************************************************************
## Printing dataframe to show all the details
#*******************************************************************************************************************************************************

# Saving the dataframe to a csv file
df.to_csv(output_dir_path + 'resume_summary_and_score.csv', index=False)


print("\n============================================================== CANDIDATE DETAILS =================================================================\n")
print(df.head(100))
print("\n============================================================== CANDIDATE DETAILS =================================================================\n")


put_text("\n\n")
put_text("**************************************************************************************************")
put_text("\tCaptured all the information Successfully !!\n\n")
put_text("\tPlease check the resume scoring and summary csv file on below path!!")
put_text("\tPath :: ", output_dir_path + 'resume_summary_and_score.csv')
put_text("**************************************************************************************************")


