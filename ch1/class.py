from unicodedata import name


class Man:
    def __init__(self,name):
        self.name = name
        print("Initailized!")

    def hello(self):
        print("hello "+self.name+"!")

    def goodbye(self):
        print("Good-bye "+self.name+"!")


m = Man("mg")
m.hello()
m.goodbye()