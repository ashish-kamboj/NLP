
from pydantic import BaseModel

## Class which describes Information required to create a MentorMentee connection

class MentorMenteeDetails(BaseModel):
    mentee_major: str = None 
    mentee_help_topics: str = None 
    mentee_experitse: str = None 
    mentor_major: str = None
    mentor_help_topics: str = None
    mentor_experitse: str = None
