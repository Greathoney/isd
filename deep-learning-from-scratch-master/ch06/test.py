def a(msg):
    print("a", msg)

def b(msg):
    print("b", msg)

def c(msg):
    print("c", msg)

messages = [a, b, c]
messages[0]("hello")