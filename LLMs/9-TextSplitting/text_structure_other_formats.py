from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

text =""" 
class Dog:
   #A simple representation of a dog.
    # A class attribute (shared by all instances)
    species = "Canis familiaris"

    def __init__(self, name, age):
        #Initializes the dog's name and age attributes
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    def bark(self):
        #Simulates a dog barking, using the dog's name
        return f"{self.name} says Woof!"

    def describe(self):
        #Provides a description of the dog
        return f"{self.name} is a {self.age}-year-old {self.species}."

# Standalone function (defined with def outside a class)
def greeting_function(name):
    # A simple function to greet a person.
    return f"Hello, {name}!"

# --- Usage Examples ---

# 1. Use the standalone function
print(f"Function output: {greeting_function('Alice')}")

# 2. Create an instance (object) of the class
my_dog = Dog("Buddy", 3)

# 3. Access attributes of the instance
print(f"Dog's name: {my_dog.name}")
print(f"Dog's species (class attribute): {Dog.species}")

# 4. Call methods of the instance
print(f"Method output 1: {my_dog.bark()}")
print(f"Method output 2: {my_dog.describe()}")


"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])