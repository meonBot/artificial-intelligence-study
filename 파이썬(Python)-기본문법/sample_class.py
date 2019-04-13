class Test : 
    def __init__(self, name):
        self.name = name
        print("Initialized!")
    
    def hello(self):
        print("Hello " + self.name + " !!")
    
    def goodBye(self):
        print("Good-Bye " + self.name +" ~~")

testClass = Test("jjj")
testClass.hello()
testClass.goodBye()

