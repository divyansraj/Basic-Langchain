from pydantic import BaseModel,Field

class Student(BaseModel):
    name: str = Field(default="Unknown", description="The name of the student")
    age: int = Field(default=0, description="The age of the student")
    email: str = Field(default="unknown@example.com", description="The email address of the student")
    cgpa: float = Field(default=0.0, ge=0.0, le=10.0, description="The CGPA of the student on a scale of 0 to 10")

new_student = Student(name='Alice', age=20, email='alice@example.com', cgpa=3.8)
print(new_student)