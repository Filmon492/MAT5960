#oppgave 18 
"""Lag et program som skriver ut alle oddetallene mellom 0 og 100."""

for i in range(0,101):
    if i % 2 == 1: # Sjekker om i er oddetall
        print(i)


#oppgave 19 
"""" Skriv et program som lager en verditabell for f(x) = x^2+3x-1
for x-verdier fra og ed 0 til og med 10 """

for x in range (0,11):
    y= x**2 + 3*x -1 # beregner y = f(x) for hver x
    print(x,y)

# oppgave 20 
""" Skriv et program som gir denne outputen:
x
xxx
xxxxx
xxxxxxx
"""
for i in range(1,8,2): # her er i lik 1,3,5,7
    print("x"*i)

