from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = "Amandeep Singh"
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt = 0, lt = 0, default = 5, description = "A decimal value representing the cgpa of the student")

new_student = {
    "age" : 22,
    "email" : "amandeep@gmail.com"
}

student = Student(**new_student) # Python dictionary unpacking

print(student) # it is an object

student_dict = dict(student) # we can convert it to dict()
print(student_dict)
print(student_dict["name"])

student_json = student.model_dump_json() # we can convert to json also
print(student_json)