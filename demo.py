def gen_primes(N):
    """Generate primes up to N"""
    primes = set()
    for n in range(2, N):
        if all(n % p > 0 for p in primes):
            primes.add(n)
            yield n

print(*gen_primes(100))


def Fibo(n):
    """Generate fibonacci numbers of given value n"""
    
    a = 1
    b = 0
    while a<n:
        print(a, end=' ')
        a,b = a+b, a
        #print(a, end=' ')
print(Fibo(1000))
