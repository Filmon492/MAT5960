# While-løkker
"""" Vi bruker while-løkker når vi ikke vet hvor mange ganger 
en operasjon skal gjentas. """

# Eksempel: 
x = 1 
while x < 10:
    #print(x)
    x = x + 1 
#print('While-løkka er ferdig')

# Eksempel 2: 
x = 1 
while x < 10:
    x = x + 1 
    #print(x)
#print('While-løkka er ferdig')

""" Snakk med sidemannen: Hva er forskjellen på de to kodesnuttene? """

# OBS: noen ganger kan det gå galt med while-løkker. 
# Eksempel: Hvorfor stopper ikke koden under? 
k = 1 
while k < 10:
    k = k +1
    #print(k)
#print('While-løkka er ferdig')

# Oppgaver: gjør oppgave 7 i heftet sammmen med sidemannen,
# vi går gjennom oppgaven i plenum etterpå. 

# Oppgaver: fortsett med oppgavene i hetftet, vi går gjennom
# et par av de dersom vi får tid. 


# oppgave 8 
""" Skrive et program som løser likningen 2^n = 128 ved å telle hvor mange ganger må vi opphøye
2 for å få 128 """
n = 0
while 2**n <= 128:
    print(n,2**n)
    n = n+1

# oppgave 9
"""Hva gjør koden?"""
partall = 0 
while partall <=10:
    if partall != 0:
        print(partall)
    partall = partall + 2
