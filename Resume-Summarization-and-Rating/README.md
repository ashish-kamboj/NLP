## Background
Need to Design a Resume Parser which will provide the resume summary (having information like Name, City, Phone No, Email etc.) and assign the rating to the resumes based on the job role to ease the resume filtration process.

## Problem
For each of the job roles we get 100’s of resumes, going through each resume and finding out the relevant one is a time consuming process which will take months for closing a single position which will lead to the scarcity of resources , which in turn causes work to be delayed or more work pressure on existing employees.

## Requirements and Phases

|                                                                          | Requirements
|--------------------------------------------------------------------------|:-----------------------------------------------------------------------------
|**Phase 1:** Regex based parser and resume rating based on few set of columns |Should have all the functions defined to extract out the candidate details
|                                                                              |Explicitly define the skills based on which we have to rate the resume
|--------------------------------------------------------------------------|-----------------------------------------------------------------------------
|**Phase 2:** Mix of Regex and ML Model based approach for Name Entity Recognition. Also, Rule based approach for resume rating based on the provided inputs                                                                          |Must have the training data (scrapping out the different resumes or trying out the public dataset available)                                                                         
|                                                                          |Build a NLP model for Name Entity Recognition
|                                                                          |Must fill out the gaps (Information didn’t extracted using ML model) using the regex functions
|                                                                          |Must have a front-end which specify all the input details (resumes input for rating, keywords file i.e. a snapshot of job description, NLP model path, skills priority and consideration)
|--------------------------------------------------------------------------|-----------------------------------------------------------------------------
|**Phase 3:** Fully ML based approach for resume summarization by identifying the key details and rate resumes based on the job description and resume text matching                                                                              |Prepare or Extract out more train data (data from different types of resumes) to build full fledged NLP model for Name Entity Recognition
|                                                                          |Try out different algorithms for Name Entity classification and choose the best one
|                                                                          |Compare the Job description text with the Resume text to generate the similarity score
|                                                                          |Enhanced front-end for including further inputs like uploading job description doc


