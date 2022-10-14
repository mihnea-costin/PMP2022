def oglindit(number):
    ogl = 0
    while number > 0:
        uc = number % 10
        ogl = ogl * 10 + uc
    return ogl


number = int(input("Introduceti numarul:\n"))
invers = oglindit(number)

print(number)