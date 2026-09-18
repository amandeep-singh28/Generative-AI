from typing import TypedDict

class Person(TypedDict):
    name : str # key 1 -> name
    age : int # key 2 -> age

new_person : Person = {
    "name" : "Amandeep",
    "age" : 22
}
new_person2 : Person = {
    "name" : "Amandeep",
    "age" : "22" # No validation
}
print(new_person)
print(new_person2)